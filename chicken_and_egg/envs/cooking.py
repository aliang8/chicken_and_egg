import itertools
import random
from functools import partial

import gymnasium as gym
import numpy as np
import pygame
import torch


class Cooking(gym.Env):
    def __init__(
        self,
        num_cells=3,
        num_steps=20,
        num_ingredients=4,
        dense_rewards=False,
        visualize=False,
        dynamics="Uniform",
        reset_recipe_between_rollouts=False,
        penalize_wrong_fridge_pickup=False,
    ):
        super(Cooking, self).__init__()
        del visualize  # Unused for this environmnent

        self.dynamics = dynamics

        self.num_cells = num_cells
        self.dense_rewards = dense_rewards
        self.num_states = num_cells**2
        self.num_ingredients = num_ingredients
        self.reset_recipe_between_rollouts = reset_recipe_between_rollouts
        self.penalize_wrong_fridge = penalize_wrong_fridge_pickup

        self._max_episode_steps = num_steps
        self.step_count = 0
        self._step = 0

        # self.font = pygame.font.SysFont(None, 18)

        self.action_names = {
            0: "0 - Up",
            1: "1 - Right",
            2: "2 - Down",
            3: "3 - Left",
            4: "4 - Pick",
            5: "5 - Drop",
        }

        self.bayes_optimal_returns = 2
        self.minimum_returns = -8

        self.task_dim = 3
        self.belief_dim = self.num_ingredients**self.num_cells

        # Possible starting states
        self.starting_state = np.array([0, 0, 0, 4, 4, 4, 4, 3, 3])

        # Goals can be anywhere except on possible starting states and
        # immediately around it
        if self.dynamics == "Uniform":
            self.possible_tasks = self.possible_goals = list(
                itertools.product(range(self.num_ingredients), repeat=3)
            )
        elif self.dynamics == "Simple_Mu":
            self.possible_tasks = self.possible_goals = list(
                itertools.product(range(self.num_ingredients // 2), repeat=3)
            )
        elif self.dynamics == "Recipe" or self.dynamics == "Vanilla":
            self.possible_tasks = self.possible_goals = [(1, 2, 3)]
        else:
            raise ValueError("Invalid dynamics: {}".format(self.dynamics))

        self.possible_tasks_np = np.array(self.possible_tasks)

        self.possible_task_to_id_map = {
            task: idx for idx, task in enumerate(self.possible_tasks)
        }

        self.task_dim = 3
        self.num_tasks = len(self.possible_tasks)

        # reset the environment state
        self._env_state = np.array(self.starting_state)
        # reset the goal
        self._task = self.reset_task()
        # reset the belief
        self._belief_state = self._reset_belief()

    @property
    def action_space(self):
        # up, right, down, left, pick up, drop off
        return gym.spaces.Discrete(6)

    @property
    def observation_space(self):
        # obs_space = [x-pos, y-pos, status, inventory, bowl ingredient 1, bowl ingredient 2, bowl ingredient 3, recipe 1, recipe 2]
        return gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array(
                [self.num_cells - 1, self.num_cells - 1, 1, 4, 4, 4, 4, 3, 3]
            ),
        )

    def render(self, mode):
        self.CELL_SIZE = 100
        self.margin = 20

        pygame.init()
        self.w = 3 * self.CELL_SIZE + 2 * self.margin
        self.h = 3 * self.CELL_SIZE + 8 * self.margin

        # color
        WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (204, 204, 204)
        self.GREEN = (0, 200, 0)
        # RED = (200, 0, 0)
        # BLUE = (0, 0, 255)
        self.INGREDIENT_COLORS = {
            0: (204, 204, 204),  # Gray
            1: (255, 199, 44),  # Yellow
            2: (153, 27, 30),  # RED
            3: (0, 0, 255),  # BLUE
            4: (255, 255, 255),  # White (Empty)
        }
        ACTOR_COLOR = (0, 0, 0)

        # setup screen
        screen = pygame.Surface((self.w, self.h))
        screen.fill(WHITE)

        # SIZES
        self.FRIDGE_SIZE = 70
        INVENTORY_SIZE = 20
        self.bars_x = self.w - self.margin - 4 * self.CELL_SIZE
        self.bars_y = self.h - self.margin - 2.2 * self.CELL_SIZE
        self.bar_height = 0.7 * self.CELL_SIZE
        self.bar_width = 20
        self.bar_gap = 5

        self.BORDER_WIDTH = 2

        self.draw_grid(screen)

        self.draw_fridges(screen, self._task)
        self.draw_circle(  # Draw Bowl
            screen,
            self.GRAY,
            1,
            1,
            radius=35,
        )
        self.draw_recipe(screen, self.recipe, INVENTORY_SIZE)
        self.draw_circle(  # Draw Actor
            screen,
            ACTOR_COLOR,
            self._env_state[0].item(),
            self._env_state[1].item(),
            radius=20,
        )

        self.draw_inventory(screen, self._env_state[3], INVENTORY_SIZE)

        return np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))

    def draw_fridges(self, screen, ingredients):
        for i in range(3):
            self.draw_rect_center_cell(
                screen,
                self.INGREDIENT_COLORS[ingredients[i]],
                2,
                i,
                self.FRIDGE_SIZE,
            )

    def draw_rect_center_cell(
        self, screen, color, col, row, shape_size, border_width=0
    ):
        bottom_left_grid_x = self.w - self.margin - (3) * self.CELL_SIZE
        bottom_left_grid_y = self.h - self.margin - self.CELL_SIZE
        x = (
            bottom_left_grid_x
            + col * self.CELL_SIZE
            + (self.CELL_SIZE - shape_size) // 2
        )
        y = (
            bottom_left_grid_y
            - row * self.CELL_SIZE
            + (self.CELL_SIZE - shape_size) // 2
        )
        pygame.draw.rect(
            screen, color, pygame.Rect(x, y, shape_size, shape_size), border_width
        )

    def draw_circle(self, screen, color, col, row, radius):
        bottom_left_grid_x = self.w - self.margin - (3) * self.CELL_SIZE
        bottom_left_grid_y = self.h - self.margin - self.CELL_SIZE
        x = bottom_left_grid_x + col * self.CELL_SIZE + (self.CELL_SIZE) // 2
        y = bottom_left_grid_y - row * self.CELL_SIZE + (self.CELL_SIZE) // 2
        pygame.draw.circle(screen, color, (x, y), radius)

    def draw_inventory(self, screen, inventory, shape_size, border_width=0):
        bottom_left_grid_x = self.w - self.margin - (3) * self.CELL_SIZE
        bottom_left_grid_y = self.h - self.margin - self.CELL_SIZE
        x = bottom_left_grid_x + (self._step // 2) * (self.margin + shape_size)
        y = bottom_left_grid_y - 2 * self.CELL_SIZE - self.margin - shape_size // 2
        if inventory == 4:
            return
        pygame.draw.rect(
            screen,
            self.INGREDIENT_COLORS[inventory],
            pygame.Rect(x, y, shape_size, shape_size),
        )

    def draw_recipe(self, screen, recipe, shape_size, border_width=0):
        bottom_left_grid_x = self.w - self.margin - (3) * self.CELL_SIZE
        bottom_left_grid_y = self.h - self.margin - self.CELL_SIZE
        x = bottom_left_grid_x
        y = (
            bottom_left_grid_y
            - 2 * self.CELL_SIZE
            - 2 * self.margin
            - shape_size
            - shape_size // 2
        )
        pygame.draw.rect(
            screen,
            self.INGREDIENT_COLORS[recipe[0]],
            pygame.Rect(x, y, shape_size, shape_size),
        )
        pygame.draw.rect(
            screen,
            self.INGREDIENT_COLORS[recipe[1]],
            pygame.Rect(x + self.margin + shape_size, y, shape_size, shape_size),
        )

    def draw_grid(self, screen):
        for row in range(3):
            for col in range(3):
                pygame.draw.rect(
                    screen,
                    self.BLACK,
                    pygame.Rect(
                        self.w - self.margin - (row + 1) * self.CELL_SIZE,
                        self.h - self.margin - (col + 1) * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE,
                    ),
                    1,
                )

        pygame.draw.rect(
            screen,
            self.BLACK if self._step != 4 else self.GREEN,
            pygame.Rect(
                self.w - self.margin - (3) * self.CELL_SIZE,
                self.h - self.margin - (3) * self.CELL_SIZE,
                3 * self.CELL_SIZE,
                3 * self.CELL_SIZE,
            ),
            2,
        )

    def set_task(self, task):
        self._task = np.array(task)
        self.recipe = self.pick_recipe()

    def reset_task(self, task=None, holdout_indices=None) -> np.ndarray:
        """
        Reset current task (i.e. fridge ingredients).

        Returns:
            Coordinates of new task id.
        """
        # If task is a scalar (int, float, numpy.int64, etc.), use it as a seed
        if isinstance(task, (int, float, np.int64, np.float64)):
            seed = task
            # Create random number generator with seed
            rng = np.random.RandomState(seed)
            # Select a random index
            index = rng.randint(0, len(self.possible_tasks))
            while holdout_indices is not None and index in holdout_indices:
                index = rng.randint(
                    0, len(self.possible_tasks)
                )  # Recurse until a non-holdout index is found
            # Use the index to select from the tuple
            self._task = self.possible_tasks[index]

            self.recipe = self.pick_recipe(rng)
        elif task is None:
            index = random.choice(list(range(len(self.possible_tasks))))
            while holdout_indices is not None and index in holdout_indices:
                index = rng.randint(
                    0, len(self.possible_tasks)
                )  # Recurse until a non-holdout index is found
            self._task = np.array(self.possible_tasks[index])
            self.recipe = self.pick_recipe()
        else:
            raise ValueError("Invalid task: {} of type {}".format(task, type(task)))
        self._task = np.array(self._task)

        self.task = self.get_task()
        index = self.possible_tasks.index(tuple(self.task))
        self.task_id = np.array(index)

        self._env_state[-2:] = self.recipe
        self.starting_state[-2:] = self.recipe
        self._step = 0

        self._reset_belief()
        return self._task

    def pick_recipe(self, rng=None):
        if (
            self.dynamics == "Uniform"
            or self.dynamics == "Recipe"
            or self.dynamics == "Simple_Mu"
        ):
            if rng is None:
                recipe = np.array(random.choices(self._task, k=2))
            else:
                recipe = np.array(rng.choice(self._task, size=2))
        elif self.dynamics == "Vanilla":
            recipe = np.array([self._task[0], self._task[1]])
        else:
            raise ValueError("Invalid dynamics: {}".format(self.dynamics))

        return recipe

    def _reset_belief(self) -> np.ndarray:
        """
        Reset oracle belief

        Returns:
            TODO

        """
        # -1 represents unknown ingredient
        self._belief_state = np.full(
            len(self.possible_tasks), 1 / len(self.possible_goals)
        )
        return self._belief_state

    def update_belief(self, state, action) -> np.ndarray:
        """
        Update oracle belief. Given state and current action, update the belief
        about the identity of the current task (i.e. goal coordinates).

        Args:
            state: current state.
            action: agent action at current state.

        Returns:
            TODO
        """

        # obs_space = [x-pos, y-pos, status, inventory,
        #              bowl ingredient 1, bowl ingredient 2, bowl ingredient 3,
        #              recipe 1, recipe 2]
        if state[2] == 2 and action == 4:  # if at fridge and action is pickup
            item_index = state[1]
            item_value = self._task[state[1]]
        else:  # No new information is discovered
            return self._belief_state

        mask = self.possible_tasks_np[:, item_index] == item_value

        # Filter probabilities based on the mask
        self._belief_state = self._belief_state * mask

        # Re-normalize the probabilities
        self._belief_state /= self._belief_state.sum()

        return self._belief_state

    def get_task(self):
        """
        Get current task, i.e. coordinates of current goal state.
        """
        return self._task.copy()

    def get_belief(self):
        """
        Get current task, i.e. a vector defining the PMF over all possible
        goal locations
        """
        return self._belief_state.copy()

    def reset(self):
        """
        Reset the environment (but not the task).
        """
        self.step_count = 0
        self._step = 0
        self._env_state = self.starting_state

        if self.reset_recipe_between_rollouts:
            self.recipe = self.pick_recipe()
        return torch.Tensor(self._env_state), {"reward": 0}

    def meta_reset(self):
        return self.reset()

    def step(self, action):
        """
        Take step, i.e. performing a state transition given the current action,
        and performing other functions like checking if agent has reached the
        goal, and computing the reward (among others).
        """

        if isinstance(action, np.ndarray) and action.ndim == 1:
            action = action[0]
        assert self.action_space.contains(action)

        done = False

        # Perform state transition
        if self._step != 4:
            reward = -0.1
        else:
            reward = 0
        """
        Take an action in the environment and returning agent's new state.
        """
        # up, right, down, left, pick up, drop off
        if action == 0:  # up
            self._env_state[1] = min([self._env_state[1] + 1, self.num_cells - 1])
        elif action == 1:  # right
            self._env_state[0] = min([self._env_state[0] + 1, self.num_cells - 1])
        elif action == 2:  # down
            self._env_state[1] = max([self._env_state[1] - 1, 0])
        elif action == 3:  # left
            self._env_state[0] = max([self._env_state[0] - 1, 0])
        elif action == 4:  # pick up
            if self._env_state[0] == 2:  # If sitting on fridge
                self._env_state[3] = self._task[
                    self._env_state[1]
                ]  # gain item in inventory
            reward = self.reward("Pickup")
        elif action == 5:  # drop off
            if self._step % 2 == 0:
                pass
            elif np.array_equal(self._env_state[:2], np.array([1, 1])) and (
                self._env_state[3] == self.recipe[(self._step - 1) // 2]
            ):  # If sitting on bowl and holding correct item
                self._env_state[(self._step - 1) // 2 + 4] = self._env_state[3]

            self._env_state[3] = 4  # drop item in inventory

            reward = self.reward("Drop")

        # update status bit
        if self._env_state[0] == 2:
            self._env_state[2] = 2  # at the fridge
        elif np.array_equal(
            self._env_state[:2], np.array([1, 1])
        ):  # Check if actor is at bowl
            self._env_state[2] = 3
        else:
            self._env_state[2] = 0

        # Check if maximum step limit is reached
        self.step_count += 1
        if self.step_count >= self._max_episode_steps:
            done = True

        # Update ground-truth belief
        self.update_belief(self._env_state, action)

        task = self.task
        task_id = self.task_id

        event = None
        last_step = self._step - 1
        if reward > 0:
            if last_step == 0:
                event = "Picked up first ingredient"
            elif last_step == 1:
                event = "Dropped off first ingredient"
            elif last_step == 2:
                event = "Picked up second ingredient"
            elif last_step == 3:
                event = "Dropped off second ingredient"
        elif reward < -0.1:
            if last_step % 2 == 1 and action == 5:
                event = "Dropped out of bowl"
            elif last_step % 2 == 1 and action == 4:
                event = "Picked up wrong ingredient"

        info = {
            "reward": reward,
            "task": task,
            "task_id": task_id,
            "belief": self.get_belief(),
            "event": event,
            "step": self._step,
            "success": True if self._step == 4 else False,
        }

        return torch.Tensor(self._env_state), reward, done, False, info

    def reward(self, action):
        reward = -0.1
        # if on FIRST STEP - pick up first ingredient
        if self._step == 0:
            if self._env_state[3] == self.recipe[0]:  # if inventory = first ingredient
                self._step += 1
                reward = 0.25
            elif (
                action == "Pickup" and self._env_state[3] != self.recipe[0]
            ):  # if inventory = first ingredient and actor picked
                if self.penalize_wrong_fridge:
                    reward -= 0.25

        # if on SECOND STEP - drop off first ingredient
        elif self._step == 1:
            if (
                self._env_state[4].item() == self.recipe[0]
            ):  # Dropped ingredient in the bowl
                self._step += 1
                reward = 0.25
            elif (
                self._env_state[3] != self.recipe[0]
            ):  # Dropped ingredient somewhere other than the bowl
                self._step -= 1
                reward -= 0.35

        # if on THIRD STEP - pick up second ingredient
        elif self._step == 2:
            if self._env_state[3] == self.recipe[1]:  # If picked up second ingredient
                self._step += 1
                reward = 0.25
            elif (  # If picked up wrong ingredient
                action == "Pickup" and self._env_state[3] != self.recipe[1]
            ):
                if self.penalize_wrong_fridge:
                    reward -= 0.25

        # if on FOURTH STEP - drop off second ingredient
        elif self._step == 3:
            if (
                self._env_state[5].item() == self.recipe[1]
            ):  # If both ingredients are in the bowl
                reward = 0.25
                self._step += 1
            elif self._env_state[3] != self.recipe[1]:  # If dropped wrong ingredient
                self._step -= 1
                reward -= 0.35

        return reward

    @property
    def task_to_id_fn(self):
        return partial(
            self.task_to_id_torch, task_to_id_map=self.possible_task_to_id_map
        )

    @staticmethod
    def task_to_id_torch(task, task_to_id_map):
        assert type(task) == torch.Tensor

        # tasks is either [B, 3]
        dim = task.shape

        if len(dim) == 3:
            # flatten
            lead_dims = dim[:-1]
            task = task.flatten(0, 1)

        # first convert to numpy
        task = task.cpu().numpy()

        _id = [task_to_id_map[tuple(t)] for t in task]

        # convert back to torch
        _id = torch.from_numpy(np.array(_id)).to(DeviceConfig.DEVICE)

        if len(dim) == 3:
            _id = _id.view(*lead_dims)

        return _id


if __name__ == "__main__":
    env = Cooking(dynamics="Vanilla")

    obs = env.reset()
    print(obs)

    actions = [1, 1, 4, 0, 3, 5, 1, 4, 3, 5, 0, 1]
    for i in range(12):
        action = actions[i]
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            break
