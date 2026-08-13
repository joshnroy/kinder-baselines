"""Tests for tossing state_abstractions.py."""

import kinder
import numpy as np
from conftest import MAKE_VIDEOS  # pylint: disable=import-error
from gymnasium.wrappers import RecordVideo
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from relational_structs import GroundAtom, ObjectCentricState

from kinder_models.dynamic3d.tossing.parameterized_skills import (
    create_lifted_controllers,
)
from kinder_models.dynamic3d.tossing.state_abstractions import (
    InGoalRegion,
    Tossing3DStateAbstractor,
)


def _get_robot_from_state(state: ObjectCentricState):
    """Helper to get robot object from state by type."""
    robots = state.get_objects(MujocoTidyBotRobotObjectType)
    assert len(robots) == 1, f"Expected 1 robot, got {len(robots)}"
    return list(robots)[0]


def test_tossing3d_state_abstraction():
    """Tests for Tossing3DStateAbstractor()."""
    kinder.register_all_environments()
    num_objects = 1
    env = kinder.make(f"kinder/Tossing3D-o{num_objects}-v0", render_mode="rgb_array")
    if MAKE_VIDEOS:
        env.unwrapped._object_centric_env.set_render_camera("task_view")  # type: ignore # pylint: disable=protected-access
        env = RecordVideo(
            env,
            "unit_test_videos",
            name_prefix="Tossing3D-state-abstraction",
        )
    sim = env.unwrapped._object_centric_env  # pylint: disable=protected-access
    abstractor = Tossing3DStateAbstractor(sim)

    # The state abstraction in the initial state.
    obs, _ = env.reset(seed=125)
    state = env.observation_space.devectorize(obs)
    assert isinstance(state, ObjectCentricState)
    abstract_state = abstractor.state_abstractor(state)
    robot = _get_robot_from_state(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(HandEmpty {robot.name}), "
        f"(OnGround cube_0), "
        f"(Reachable cube_0 cuboid_barrier)]"
    )

    # The goal is the environment's own success criterion.
    cube = state.get_object_from_name("cube_0")
    goal = abstractor.goal_deriver(state)
    assert goal.atoms == {GroundAtom(InGoalRegion, [cube])}
    assert not goal.check_state(state)
    assert not sim._check_goals()  # pylint: disable=protected-access

    # The region is the task JSON's "ranges" inflated by the ground placement threshold
    # (0.05 m). x = 2.14 is outside the literal 2.10 but inside the inflated 2.15,
    # which is what tells the two apart.
    landed_state = state.copy()
    landed_state.set(cube, "x", 2.14)
    landed_state.set(cube, "y", 0.0)
    landed_state.set(cube, "z", 0.025)
    landed_atoms = abstractor.state_abstractor(landed_state).atoms
    assert GroundAtom(InGoalRegion, [cube]) in landed_atoms

    # Drive the base to a throw standoff from the bin, so NearBin is checked against
    # the controller that establishes it rather than only against the initial state.
    controllers = create_lifted_controllers(env.action_space)
    lifted_controller = controllers["move_to_target"]
    target_bin = state.get_object_from_name("bin_0")
    object_parameters = (robot, target_bin)
    controller = lifted_controller.ground(object_parameters)
    throw_standoff = 1.35
    # cube_0 rests at x = 0.71 and the base is headed for x = 0.65, so it is excluded
    # the way MoveToThrowPoseController excludes the object it holds.
    controller.reset(
        state,
        np.array([throw_standoff, 0.0]),
        disable_collision_objects=["cube_0"],
    )
    for _ in range(300):
        action = controller.step()
        obs, _, _, _, _ = env.step(action)
        next_state = env.observation_space.devectorize(obs)
        controller.observe(next_state)
        state = next_state
        if controller.terminated():
            break
    else:
        assert False, "Controller did not terminate"

    # Standing where it can throw from, with nothing else changed.
    abstract_state = abstractor.state_abstractor(state)
    assert str(sorted(abstract_state.atoms)) == (
        f"[(HandEmpty {robot.name}), "
        f"(NearBin {robot.name} bin_0), "
        f"(OnGround cube_0), "
        f"(Reachable cube_0 cuboid_barrier)]"
    )

    env.close()
