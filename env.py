import gymnasium as gym 
from gymnasium import spaces
import pygame
import numpy as np 

class HanoiEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, n_disks, render_mode=None):
        assert 3 <= n_disks <= 12, f"Number of disks must be between 3 and 12"
        self.n_disks = n_disks
        
        # our observation is what is on each tower
        # each tower has a list of n elements which can be 0 or 1
        # 0 means the disk at that position is absent, 1 means it is present
        self.observation_space = spaces.Dict(
            {
                "tower 1": spaces.MultiBinary(self.n_disks),
                "tower 2": spaces.MultiBinary(self.n_disks),
                "tower 3": spaces.MultiBinary(self.n_disks),
            }
        )
        self.towers = np.zeros((3, self.n_disks))
        # our action is a pair(source,destination)
        # move top disk on source to destination 
        self.action_space = spaces.Discrete(6)
        self.action_to_src_des = {
            0: (0, 1),
            1: (0, 2),
            2: (1, 0),
            3: (1, 2),
            4: (2, 0),
            5: (2, 1)
        }
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None 
        self.clock = None
        
        # Rendering constants
        self.window_width = 800
        self.window_height = 600
        self.tower_spacing = self.window_width // 4
        self.tower_base_y = self.window_height - 100
        self.disk_height = 20
        self.max_disk_width = 150
        self.min_disk_width = 30
        
    def _get_obs(self):
        return {"tower 1": self.towers[0], "tower 2": self.towers[1], "tower 3": self.towers[2]} 
    
    def _get_info(self):
        return {
            "remaining": int(self.n_disks - np.sum(self.towers[2]))
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.towers[0] = np.ones(self.n_disks)
        self.towers[1] = np.zeros(self.n_disks)
        self.towers[2] = np.zeros(self.n_disks)
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info
    
    def step(self, action):
        # action is a pair(source, destination)
        a = self.action_to_src_des[action]
        src = a[0]
        des = a[1]
        for disk in range(self.n_disks):
            # find smallest disk on source
            if self.towers[src][disk] == 0:
                continue 
            # found smallest disk. check if there is no disk smaller than that on destination
            invalid = False 
            for i in range(disk):
                if self.towers[des][i] == 1:
                    # this is smaller than the src disk and should not be allowed
                    invalid = True
                    break 
            # if invalid do not change env state 
            if not invalid:
                self.towers[src][disk] = 0
                self.towers[des][disk] = 1
            break
        obs = self._get_obs()
        info = self._get_info()
        truncated = False
        terminated = np.sum(self.towers[2]) == self.n_disks
        reward = -1  # -1 for every time step
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != "human":
            return
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Tower of Hanoi")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Fill background
        self.window.fill((255, 255, 255))
        
        # Draw towers (poles)
        tower_positions = [
            self.tower_spacing,
            self.tower_spacing * 2,
            self.tower_spacing * 3
        ]
        
        for tower_x in tower_positions:
            # Draw pole
            pygame.draw.rect(
                self.window,
                (100, 100, 100),
                (tower_x - 5, self.tower_base_y - 300, 10, 300)
            )
            # Draw base
            pygame.draw.rect(
                self.window,
                (150, 150, 150),
                (tower_x - 100, self.tower_base_y, 200, 20)
            )
        
        # Draw disks
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 100),  # Orange
            (150, 100, 255),  # Purple
            (255, 200, 100),  # Light orange
            (100, 200, 255),  # Light blue
            (200, 255, 100),  # Light green
            (255, 100, 200),  # Pink
        ]
        
        for tower_idx in range(3):
            tower_x = tower_positions[tower_idx]
            disk_position = 0  # Position from bottom of tower
            
            # Draw disks from largest to smallest (bottom to top)
            for disk_size in range(self.n_disks - 1, -1, -1):
                if self.towers[tower_idx][disk_size] == 1:
                    # Calculate disk width based on size
                    disk_width = self.min_disk_width + (self.max_disk_width - self.min_disk_width) * (disk_size / (self.n_disks - 1))
                    
                    # Calculate y position
                    y_pos = self.tower_base_y - self.disk_height * (disk_position + 1)
                    
                    # Draw disk
                    pygame.draw.rect(
                        self.window,
                        colors[disk_size % len(colors)],
                        (tower_x - disk_width / 2, y_pos, disk_width, self.disk_height)
                    )
                    # Draw disk border
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 0),
                        (tower_x - disk_width / 2, y_pos, disk_width, self.disk_height),
                        2
                    )
                    
                    disk_position += 1
        
        # Draw tower labels
        font = pygame.font.Font(None, 36)
        for i, tower_x in enumerate(tower_positions):
            label = font.render(f"Tower {i}", True, (0, 0, 0))
            label_rect = label.get_rect(center=(tower_x, self.tower_base_y + 50))
            self.window.fill((255, 255, 255), label_rect)
            self.window.blit(label, label_rect)
        
        # Draw move counter
        info = self._get_info()
        moves_text = font.render(f"Remaining: {info['remaining']}", True, (0, 0, 0))
        self.window.blit(moves_text, (10, 10))
        
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
