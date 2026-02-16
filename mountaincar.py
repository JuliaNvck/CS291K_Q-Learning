import numpy as np
import pygame
from pygame import gfxdraw


class MountainCar:
    """Adaptation of the Gymnasium Mountain Car environment for this course.

    Changes:
    - Maintains an array of environments for much faster stepping.
    - Removed code supporting image observations.
    - Added a dense reward option.
    - Added "reseed" method to reset PRNG.

    For details, see the handout and the original API at:
    https://gymnasium.farama.org/environments/classic_control/mountain_car/
    """

    def __init__(self, n_envs, sparse_reward=True, seed=0):
        """Initializes the MountainCar environment.

        Args:
            n_envs (int): Number of parallel environments to maintain.
            sparse_reward (bool): If True, use sparse reward (1 at goal). If False,
                use dense reward that includes progress-based shaping.
            seed (int): Random seed for reproducibility.
        """
        self.n_envs = n_envs
        self.state = None
        self.sparse = sparse_reward
        self.N_ACTIONS = 3

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.force = 0.001
        self.gravity = 0.0025

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.rng = np.random.default_rng(seed=seed)

    def reseed(self, seed):
        """Reseeds the random number generator for reproducibility.

        Args:
            seed (int): New random seed.
        """
        self.rng = np.random.default_rng(seed=seed)

    def step(self, actions):
        """Advances all environments by one time step using the given actions.

        Args:
            actions (array(N) of int): Actions for each environment. Must be in {0, 1, 2}
                where 0=push left, 1=no push, 2=push right.

        Returns:
            (states, rewards, is_terminals): Tuple where:
                states (array(N, 2)): New states [position, velocity] for each environment.
                rewards (array(N)): Rewards received for each environment.
                is_terminals (array(N) of bool): Whether each environment reached the goal.
        """
        valid = (actions == 0) | (actions == 1) | (actions == 2)
        if not np.all(valid):
            raise ValueError("actions must be integers {0, 1, 2}.")

        positions, velocities = self.state.T

        # repeat the action twice.
        for _ in range(2):
            velocities += (actions - 1) * self.force + np.cos(3 * positions) * (-self.gravity)
            velocities = np.clip(velocities, -self.max_speed, self.max_speed)
            positions += velocities
            positions = np.clip(positions, self.min_position, self.max_position)

            mask_left_barrier = (positions == self.min_position) & (velocities < 0)
            velocities[mask_left_barrier] = 0

        self.state = np.stack([positions, velocities]).T

        at_goal = (positions >= self.goal_position) & (velocities >= 0)
        reward = 1.0 * at_goal
        if not self.sparse:
            reward += 1.0 * np.maximum(positions + 0.5, 0) * np.maximum(velocities, 0)

        return self.state.copy(), reward, at_goal

    def reset(self, entire_state_space=False, mask=None):
        """Resets environments specified by mask to initial states.

        Args:
            entire_state_space (bool): If True, sample uniformly from entire state
                space. If False, use fixed initial position (-0.5, 0).
            mask (array(N) of bool): Boolean mask indicating which environments to
                reset. If None, resets all environments.

        Returns:
            states (array(N, 2)): States [position, velocity] for all environments
                after reset.
        """
        if mask is None:
            mask = np.repeat(True, self.n_envs)

        n_reset = np.sum(mask)

        if entire_state_space:
            pos = self.rng.uniform(self.min_position, self.max_position, size=n_reset)
            vel = self.rng.uniform(-self.max_speed, self.max_speed, size=n_reset)
        else:
            # fixed initial position.
            pos = -0.5 * np.ones(n_reset)
            vel = 0 * pos
        if self.state is None:
            assert np.all(mask)
            self.state = np.zeros((self.n_envs, 2))
        self.state[mask] = np.stack([pos, vel]).T
        return self.state.copy()

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self):
        """Renders the scene to a graphics window."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        # Always render environment 0.
        pos = self.state[0, 0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = np.stack([(xs - self.min_position) * scale, ys * scale]).T

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        pygame.event.pump()
        self.clock.tick(60)  # target FPS
        pygame.display.flip()

    def close(self):
        """Closes the rendering window and cleans up pygame resources."""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class FeatureMap:
    """Feature map for MountainCar using RBF features with cross products.

    Creates a feature representation using bias, linear, and radial basis function
    (RBF) features for position and velocity, along with all pairwise cross products.
    """

    def __init__(self):
        """Initializes the feature map with RBF centers."""
        mc = MountainCar(n_envs=1)
        self.POS_STEPS = np.linspace(mc.min_position, mc.max_position, 8)
        self.VEL_STEPS = np.linspace(-mc.max_speed, mc.max_speed, 8)

    def __call__(self, states):
        """Transforms states into feature vectors.

        Args:
            states (array(N, 2)): Batch of N states with [position, velocity].

        Returns:
            features (array(N, d)): Feature vectors where d is the feature dimension.
                Features include bias, linear terms, RBF features for position and
                velocity, and all pairwise cross products.
        """
        pos, vel = states.T
        N = len(pos)
        features = [np.ones((N, 1)), pos[:, None], vel[:, None]]
        for val, steps in zip((pos, vel), (self.POS_STEPS, self.VEL_STEPS)):
            relative = val[:, None] - steps[None, :]
            stepwidth = 1.0 * (steps[1] - steps[0])
            feature = np.exp(-((relative / stepwidth) ** 2))
            features.append(feature)
        features = np.concatenate(features, axis=-1)
        cross_features = features[:, None, :] * features[:, :, None]
        return cross_features.reshape((len(pos), -1))
