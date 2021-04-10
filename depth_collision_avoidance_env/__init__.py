from typing import Tuple

import numpy as np
import math
from collections import deque
from gym_miniworld.miniworld import MiniWorldEnv
import matplotlib.pyplot as plt
from gym_miniworld.entity import Box, Agent
from gym_miniworld.params import DEFAULT_PARAMS
from gym import spaces
from gym.envs import register
from gym_miniworld.opengl import *
from typing import List, Dict


class DepthCollisionAvoidance(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, num_agents=4, pref_speed=1.3, max_speed=1.5, delta_time=0.10,
                 depth_length=10, **kwargs):
        assert size >= 2
        self.size = size
        self.pref_speed = pref_speed
        self.max_speed = max_speed
        self.delta_time = delta_time

        self.num_agents = num_agents
        self.agents = {}
        self.playback = []

        self.observations = {}
        self.depth_length = depth_length

        super().__init__(
            obs_width=80,
            obs_height=64,
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Drive with a linear and turning force. This is pretty much polar velocity.
        self.action_space = spaces.Tuple(
            tuple(
                spaces.Box(low=np.array([0, -np.pi / 2]),
                           high=np.array([max_speed, np.pi / 2]), dtype=np.float32)
                for _ in range(num_agents)
            )
        )

        # Observe with a depth image 60x80 and the relative position of the goal
        self.observation_space = spaces.Tuple(
            tuple(
                spaces.Tuple((
                    spaces.Box(low=0, high=4.5, shape=(60, 80, 1)),
                    spaces.Box(low=np.array([0, -np.pi]),
                               high=np.array([np.sqrt(2 * size), np.pi]), shape=(2,)),
                    spaces.Box(low=np.array([0, -np.pi / 2]),
                               high=np.array([max_speed, np.pi / 2]), dtype=np.float32)
                )) for _ in range(num_agents))
        )

    def _gen_world(self):
        # Define the room
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.agents = {}
        self.observations = {}
        # Add all agents to the world
        for i in range(self.num_agents):
            # Initialize the agent
            agent = Agent()

            # Randomize the goal position of the agent
            agent.goal = np.array([np.random.rand() * self.size,
                                   0,
                                   np.random.rand() * self.size])

            # Place the agent down in the environment
            self.place_entity(
                agent,
                room=room,
                min_x=0,
                max_x=self.size,
                min_z=0,
                max_z=self.size
            )

            # Append it to the list
            self.agents[i] = agent
            self.observations[i] = deque(maxlen=self.depth_length)

            # Keep self.agent populated with an agent so that some internal functions work
            self.agent = agent

    def render_obs_agent(self, agent, frame_buffer=None):
        """
        Render an observation from the point of view of the agent
        """

        if frame_buffer == None:
            frame_buffer = self.obs_fb

        # Switch to the default OpenGL context
        # This is necessary on Linux Nvidia drivers
        self.shadow_window.switch_to()

        # Bind the frame buffer before rendering into it
        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*self.sky_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            agent.cam_fov_y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,
            100.0
        )

        # Setup the camera
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *agent.cam_pos,
            # Target
            *(agent.cam_pos + agent.cam_dir),
            # Up vector
            0, 1.0, 0.0
        )

        return self._render_world_agent(
            frame_buffer,
            agent,
            render_agent=False
        )

    def _render_world_agent(
            self,
            frame_buffer,
            agent,
            render_agent
    ):
        """
        Render the world from a given camera position into a frame buffer,
        and produce a numpy image array as output.
        """

        # Call the display list for the static parts of the environment
        glCallList(1)

        # TODO: keep the non-static entities in a different list for efficiency?
        # Render the non-static entities
        for ent in self.entities:
            if ent is not agent:
                ent.render()

    def render(self):
        img = self.render_top_view(self.vis_fb)

        img_width = img.shape[1]
        img_height = img.shape[0]

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=True)
            self.window = pyglet.window.Window(
                width=img_width,
                height=img_height,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Clear the color and depth buffers
        glClearColor(0, 0, 0, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, img_width, 0, img_height, 0, 10)

        # Draw the human render to the rendering window
        img_flip = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = pyglet.image.ImageData(
            img_width,
            img_height,
            'RGB',
            img_flip.ctypes.data_as(POINTER(GLubyte)),
            pitch=img_width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=img_width,
            height=img_height
        )

        # Force execution of queued commands
        glFlush()

        self.window.flip()
        self.window.dispatch_events()

        return img

    def observation(self):
        observations = {}

        self._render_static()

        for agent_id in self.agents:
            agent = self.agents[agent_id]
            frame_buffer = FrameBuffer(80, 64, 8)
            self.render_obs_agent(agent, frame_buffer)
            frame_buffer.resolve()
            depth = frame_buffer.get_depth_map(0.4, 4.5)
            while len(self.observations[agent_id]) < self.depth_length:
                self.observations[agent_id].append(depth)

            self.observations[agent_id].append(depth)

            goal_offset = agent.goal - agent.pos
            goal_r = np.sqrt(goal_offset[0] ** 2 + goal_offset[2] ** 2)
            goal_a = np.arctan2(goal_offset[0], goal_offset[2])
            vel_r = np.sqrt(agent.vel[0] ** 2 + agent.vel[2] ** 2)
            vel_a = np.arctan2(agent.vel[0], agent.vel[2])

            observations[agent_id] = (
                self.observations[agent_id],
                (goal_r, goal_a),
                (vel_r, vel_a)
            )

        return observations

    def reset(self):
        super(DepthCollisionAvoidance, self).reset()

        self.playback = []

        return self.observation()

    def step(self, actions: Dict[int, Tuple[float, float]], collisions: bool=True):
        rewards = {}
        info = {"agents": {}}
        done = False

        self.playback += [(id, self.agents[id].pos[0], self.agents[id].pos[2], self.step_count)
                          for id in self.agents]

        for agent_id in actions:
            action = actions[agent_id]
            # Get the linear and rotational aspects of the action
            linear = action[0]
            rot = action[1]

            agent = self.agents[agent_id]

            # Turn this into an X and Y velocity
            agent.dir += rot
            agent.dir %= 2 * np.pi
            vel = np.array(
                [np.sin(agent.dir + np.pi / 2), np.cos(agent.dir + np.pi / 2)]) * self.max_speed * linear
            vel = np.array([vel[0], 0, vel[1]])

            # Keep old position for reward
            old_pos = agent.pos

            # Check that the new position does not intersect with anything
            prop_position = agent.pos + vel * self.delta_time
            if not self.intersect(agent, prop_position, agent.radius):
                agent.pos = prop_position
                agent.vel = vel
            else:
                agent.vel = np.array([0, 0, 0])

            # Calculate the reward as the energy efficiency of the last movement
            old_goal_dist = np.linalg.norm(agent.goal - old_pos)
            new_goal_dist = np.linalg.norm(agent.goal - agent.pos)
            reward = float(old_goal_dist - new_goal_dist) / float(np.linalg.norm(vel))

            # Check for collisions and goal positions and rectify the reward
            for other_agent_id in self.agents:
                other_agent = self.agents[other_agent_id]
                if other_agent is agent:
                    continue

                if self.near(agent, other_agent, eps=0.05) and collisions:
                    reward = -15.
                    done = True
                    info["agents"][agent_id] = "collision"
                    break

            if new_goal_dist < 1:
                info["agents"][agent_id] = "goal"
                done = True
                reward = 15.

            # Add the reward to the rewards
            rewards[agent_id] = reward

        # Observe the environment
        observations = self.observation()

        # Check step counts
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            done = True
            info = {"term_reason": "max_steps"}

        # Return information
        return observations, rewards, done, info


register(
    id="DepthCollisionAvoidance-v0",
    entry_point="depth_collision_avoidance_env:DepthCollisionAvoidance",
)
