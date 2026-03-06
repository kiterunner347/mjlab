"""Tests for sim.py."""

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.sim import MujocoCfg, Simulation, SimulationCfg


@pytest.fixture
def device():
  """Test device fixture."""
  return get_test_device()


@pytest.fixture
def robot_xml():
  """Simple robot with geoms and joints."""
  return """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint name="free_joint"/>
          <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="1.0"
            friction="0.5 0.01 0.005"/>
          <body name="foot1" pos="0.2 0 0">
            <joint name="joint1" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot1_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
          <body name="foot2" pos="-0.2 0 0">
            <joint name="joint2" type="hinge" axis="0 0 1" range="0 1.57"/>
            <geom name="foot2_geom" type="box" size="0.05 0.05 0.05" mass="0.1"
              friction="0.5 0.01 0.005"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """


def test_simulation_config_is_piped(robot_xml, device):
  """Test that SimulationCfg values are applied to both mj_model and wp_model."""
  model = mujoco.MjModel.from_xml_string(robot_xml)

  cfg = SimulationCfg(
    contact_sensor_maxmatch=128,
    ls_parallel=False,
    mujoco=MujocoCfg(
      timestep=0.02,
      integrator="euler",
      solver="cg",
      iterations=7,
      ls_iterations=14,
      ccd_iterations=20,
      gravity=(0, 0, 7.5),
      multiccd=True,
    ),
  )

  sim = Simulation(num_envs=1, cfg=cfg, model=model, device=device)

  # MujocoCfg should be applied to mj_model.
  assert sim.mj_model.opt.timestep == cfg.mujoco.timestep
  assert sim.mj_model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert sim.mj_model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  assert sim.mj_model.opt.iterations == cfg.mujoco.iterations
  assert sim.mj_model.opt.ls_iterations == cfg.mujoco.ls_iterations
  assert sim.mj_model.opt.ccd_iterations == cfg.mujoco.ccd_iterations
  assert tuple(sim.mj_model.opt.gravity) == cfg.mujoco.gravity
  assert sim.mj_model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD

  # MujocoCfg should be inherited by wp_model via put_model.
  np.testing.assert_almost_equal(
    sim.model.opt.timestep[0].cpu().numpy(), cfg.mujoco.timestep
  )
  np.testing.assert_almost_equal(
    sim.model.opt.gravity[0].cpu().numpy(), cfg.mujoco.gravity
  )
  assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
  assert sim.model.opt.solver == mujoco.mjtSolver.mjSOL_CG
  assert sim.model.opt.iterations == cfg.mujoco.iterations
  assert sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD

  # SimulationCfg should be applied to wp_model.
  assert sim.wp_model.opt.contact_sensor_maxmatch == cfg.contact_sensor_maxmatch
  assert sim.wp_model.opt.ls_parallel == cfg.ls_parallel


def test_sim_reset_restores_initial_state(robot_xml, device):
  """Test that sim.reset() restores qpos/qvel to initial values."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=2, cfg=SimulationCfg(), model=model, device=device)

  qpos0 = sim.data.qpos.clone()
  qvel0 = sim.data.qvel.clone()

  # Run simulation to modify state.
  for _ in range(10):
    sim.step()

  assert not torch.allclose(sim.data.qpos, qpos0)
  assert not torch.allclose(sim.data.qvel, qvel0)

  # Reset should restore initial state.
  sim.reset()

  torch.testing.assert_close(sim.data.qpos[:], qpos0)
  torch.testing.assert_close(sim.data.qvel[:], qvel0)
  # qacc_warmstart should be zeroed.
  assert (sim.data.qacc_warmstart == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Likely bug on CPU MjWarp")
def test_sim_reset_selective(robot_xml, device):
  """Test that sim.reset() only affects specified environments."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  sim = Simulation(num_envs=4, cfg=SimulationCfg(), model=model, device=device)

  qpos0 = sim.data.qpos.clone()

  # Run simulation to modify state.
  for _ in range(10):
    sim.step()

  qpos_after_sim = sim.data.qpos.clone()

  # Reset only env 1 and 3.
  sim.reset(torch.tensor([1, 3], device=device))

  # Envs 1 and 3 should be reset.
  torch.testing.assert_close(sim.data.qpos[1], qpos0[1])
  torch.testing.assert_close(sim.data.qpos[3], qpos0[3])
  # Envs 0 and 2 should be unchanged.
  torch.testing.assert_close(sim.data.qpos[0], qpos_after_sim[0])
  torch.testing.assert_close(sim.data.qpos[2], qpos_after_sim[2])


def test_xpos_matches_qpos_after_forward(robot_xml, device):
  """sim.step() leaves xpos stale; sim.forward() makes it match qpos.

  In MuJoCo, mj_step = mj_step1 (forward kinematics + forces) + mj_step2
  (integration). After mj_step, qpos/qvel are post-integration but xpos is
  from the pre-integration forward pass. sim.forward() recomputes xpos from
  the current qpos.
  """
  model = mujoco.MjModel.from_xml_string(robot_xml)
  cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.01))  # Large dt for clear signal
  sim = Simulation(num_envs=2, cfg=cfg, model=model, device=device)

  # Step enough for significant velocity -> large staleness gap.
  for _ in range(50):
    sim.step()

  # xpos is stale: reflects pre-integration state of last step.
  # For the freejoint body (body 1), qpos[:3] is the true position.
  xpos_stale = sim.data.xpos[:, 1].clone()
  qpos_pos = sim.data.qpos[:, :3].clone()
  assert not torch.allclose(xpos_stale, qpos_pos, atol=1e-4)

  # forward() refreshes derived quantities from current qpos.
  sim.forward()
  xpos_fresh = sim.data.xpos[:, 1].clone()
  torch.testing.assert_close(xpos_fresh, qpos_pos, atol=1e-5, rtol=0)


def test_step1_step2_equivalent_to_step(robot_xml, device):
  """step1() + step2() must produce the same result as step()."""
  model = mujoco.MjModel.from_xml_string(robot_xml)
  cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.005))

  # Run step() path.
  sim_step = Simulation(num_envs=2, cfg=cfg, model=model, device=device)
  for _ in range(20):
    sim_step.step()
  qpos_step = sim_step.data.qpos.clone()
  qvel_step = sim_step.data.qvel.clone()

  # Run step1()+step2() path with fresh simulation.
  sim_two = Simulation(num_envs=2, cfg=cfg, model=model, device=device)
  for _ in range(20):
    sim_two.step1()
    sim_two.step2()
  qpos_two = sim_two.data.qpos.clone()
  qvel_two = sim_two.data.qvel.clone()

  torch.testing.assert_close(qpos_two, qpos_step, atol=1e-5, rtol=0)
  torch.testing.assert_close(qvel_two, qvel_step, atol=1e-5, rtol=0)


def test_step2_uses_ctrl_set_after_step1(device):
  """Controls written between step1() and step2() should affect dynamics."""
  model_xml = """
    <mujoco>
      <worldbody>
        <body name="arm" pos="0 0 1">
          <joint name="j1" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size="0.05 0.2" mass="1.0"/>
          <body name="forearm" pos="0 0 0.4">
            <joint name="j2" type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.04 0.15" mass="0.5"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j1" gear="10"/>
        <motor joint="j2" gear="5"/>
      </actuator>
    </mujoco>
  """
  model = mujoco.MjModel.from_xml_string(model_xml)
  cfg = SimulationCfg(mujoco=MujocoCfg(timestep=0.005))
  num_envs = 2

  # Simulation with zero controls.
  sim_zero = Simulation(num_envs=num_envs, cfg=cfg, model=model, device=device)
  for _ in range(10):
    sim_zero.step1()
    sim_zero.step2()
  qvel_zero = sim_zero.data.qvel.clone()

  # Simulation with non-zero controls set between step1 and step2.
  sim_ctrl = Simulation(num_envs=num_envs, cfg=cfg, model=model, device=device)
  ctrl_value = torch.ones(num_envs, model.nu, device=device)
  for _ in range(10):
    sim_ctrl.step1()
    sim_ctrl.data.ctrl[:] = ctrl_value
    sim_ctrl.step2()

  # Non-zero control should produce different joint velocities.
  assert not torch.allclose(sim_ctrl.data.qvel, qvel_zero, atol=1e-4)
