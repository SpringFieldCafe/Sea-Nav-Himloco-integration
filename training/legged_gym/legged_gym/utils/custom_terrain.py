import numpy as np

def is_path_with_obstacle(room, robot_pos, goal_pos):
    """Check if there is any obstacle between the robot and the goal."""
    x1, y1 = robot_pos
    x2, y2 = goal_pos
    
    # Compute line step increments
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    if dx > dy:
        err = dx / 2.0
        while x1 != x2:
            if room[x1, y1] > 0:  # Obstacle check
                return True
            err -= dy
            if err < 0:
                y1 += sy
                err += dx
            x1 += sx
    else:
        err = dy / 2.0
        while y1 != y2:
            if room[x1, y1] > 0:  # Obstacle check
                return True
            err -= dx
            if err < 0:
                x1 += sx
                err += dy
            y1 += sy
    
    return False  # No obstacles

def is_far_from_obstacles(room, pos, min_distance):
    """Check if the position is at least min_distance away from obstacles."""
    y_start = max(0, pos[0] - min_distance)
    y_end = min(room.shape[0], pos[0] + min_distance + 1)
    x_start = max(0, pos[1] - min_distance)
    x_end = min(room.shape[1], pos[1] + min_distance + 1)
    
    neighborhood = room[y_start:y_end, x_start:x_end]
    if neighborhood.size == 0:
        return False
    
    # An obstacle is defined as a cell with height > 0.1m
    no_obst = (neighborhood <= 0.1).all()
    return no_obst

def scale_robot_and_goal(robot_pos=None, goal_pos=None, scale_factor=5):
    final_goal_pos_x = goal_pos[0] * scale_factor
    final_goal_pos_y = goal_pos[1] * scale_factor
    final_goal_pos = [final_goal_pos_x, final_goal_pos_y]
    final_robot_pos_x = robot_pos[0] * scale_factor
    final_robot_pos_y = robot_pos[1] * scale_factor
    final_robot_pos = [final_robot_pos_x, final_robot_pos_y]
    return final_robot_pos, final_goal_pos
            
def place_robot_and_goal(
    room,
    min_distance=5,
    min_goal_distance=35,
    max_attempts=50_000,
):
    """Place valid robot and goal positions with bounded retries."""
    rows, cols = room.shape

    if rows < 3 or cols < 3:
        raise ValueError(
            f"Room is too small for placement: room_shape={room.shape}"
        )

    if max_attempts <= 0:
        raise ValueError(
            f"max_attempts must be greater than 0, got {max_attempts}"
        )

    # Only require the direct path to cross an obstacle when the room
    # actually contains an interior obstacle.
    interior_obstacle_cells = int(
        np.count_nonzero(room[1:-1, 1:-1] > 0.1)
    )
    require_path_obstacle = interior_obstacle_cells > 0

    rejected = {
        "occupied": 0,
        "near_obstacle": 0,
        "goal_too_close": 0,
        "path_without_obstacle": 0,
    }

    for attempt in range(1, max_attempts + 1):
        robot_pos = [
            np.random.randint(1, rows - 1),
            np.random.randint(1, cols - 1),
        ]
        goal_pos = [
            np.random.randint(1, rows - 1),
            np.random.randint(1, cols - 1),
        ]

        if (
            room[tuple(robot_pos)] != 0
            or room[tuple(goal_pos)] != 0
        ):
            rejected["occupied"] += 1

        elif (
            not is_far_from_obstacles(
                room, robot_pos, min_distance
            )
            or not is_far_from_obstacles(
                room, goal_pos, min_distance
            )
        ):
            rejected["near_obstacle"] += 1

        elif (
            np.linalg.norm(
                np.asarray(robot_pos) - np.asarray(goal_pos)
            )
            <= min_goal_distance
        ):
            rejected["goal_too_close"] += 1

        elif (
            require_path_obstacle
            and not is_path_with_obstacle(
                room, robot_pos, goal_pos
            )
        ):
            rejected["path_without_obstacle"] += 1

        else:
            return robot_pos, goal_pos

        if attempt % 5_000 == 0:
            print(
                "[place_robot_and_goal] "
                f"attempt={attempt}/{max_attempts}, "
                f"interior_obstacle_cells={interior_obstacle_cells}, "
                f"require_path_obstacle={require_path_obstacle}, "
                f"rejected={rejected}",
                flush=True,
            )

    raise RuntimeError(
        "Unable to place robot and goal after "
        f"{max_attempts} attempts; "
        f"room_shape={room.shape}, "
        f"min_distance={min_distance}, "
        f"min_goal_distance={min_goal_distance}, "
        f"interior_obstacle_cells={interior_obstacle_cells}, "
        f"require_path_obstacle={require_path_obstacle}, "
        f"rejected={rejected}"
    )

def create_room(grid_size=10):
    """Create a room with walls around the boundary."""
    room = np.zeros((grid_size, grid_size), dtype=float)
    room[0, :] = 1.0  # Top wall
    room[-1, :] = 1.0  # Bottom wall
    room[:, 0] = 1.0  # Left wall
    room[:, -1] = 1.0  # Right wall
    
    return room

def generate_random_shape(shape_size=30, num_cells=20):
    """
    Generate a random shape composed of adjacent cells.

    Args:
        shape_size: Size of the local square grid.
        num_cells: Number of cells in the shape.

    Returns:
        A local grid containing the shape.
    """
    shape = np.zeros((shape_size, shape_size))
    
    # Random start
    start_row = np.random.randint(1, shape_size - 1)
    start_col = np.random.randint(1, shape_size - 1)
    
    shape[start_row, start_col] = 1
    
    # Four possible directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Track expansion with a stack
    cells = [(start_row, start_col)]
    
    while len(cells) < num_cells:
        # Randomly choose a cell to expand
        current_row, current_col = cells[np.random.randint(len(cells))]
        
        # Randomly choose a direction
        direction = directions[np.random.randint(4)]
        new_row = current_row + direction[0]
        new_col = current_col + direction[1]
        
        # Ensure in bounds and not occupied
        if 0 <= new_row < shape_size and 0 <= new_col < shape_size:
            shape[new_row, new_col] = np.random.randint(40, 100) * 0.01 # TODO: config for the range of obstacle height, now set to 0.2m ~ 1.0m
            cells.append((new_row, new_col))
    
    return shape

def add_obstacles(room, level, grid_size=100, shape_size=3):
    """Randomly add obstacle shapes. The count increases with difficulty level."""
    num_obstacles = int(level)
    
    for _ in range(num_obstacles):
        # Reduce cluster size to avoid large walls
        num_cells = np.random.randint(2, 4) 
        shape = generate_random_shape(shape_size, num_cells)
        
        # Random placement
        start_row = np.random.randint(1, grid_size - shape_size)
        start_col = np.random.randint(1, grid_size - shape_size)
        
        # Place into the room
        room_section = room[start_row:start_row + shape_size, start_col:start_col + shape_size]
        room[start_row:start_row + shape_size, start_col:start_col + shape_size] = np.maximum(room_section, shape)
            
    for _ in range(num_obstacles*5): # Increase count of small obstacles
        shape_size = 3
        num_cells = 1 # Mostly single cells or pairs
        shape = generate_random_shape(3, num_cells)
        shape *= np.random.randint(40, 100) * 0.01
        
        # Random placement
        start_row = np.random.randint(1, grid_size - shape_size)
        start_col = np.random.randint(1, grid_size - shape_size)
        
        # Place into the room
        room_section = room[start_row:start_row + shape_size, start_col:start_col + shape_size]
        room[start_row:start_row + shape_size, start_col:start_col + shape_size] = np.maximum(room_section, shape)

    return room

def scale_room(room, scale_factor=None):
    scale_factor = int(scale_factor)
    scaled_size = int(room.shape[0] * scale_factor)
    scaled_room = np.zeros((scaled_size, scaled_size), dtype=float)
    
    for i in range(room.shape[0]):
        for j in range(room.shape[1]):
            scaled_room[i*scale_factor:(i+1)*scale_factor, j*scale_factor:(j+1)*scale_factor] = room[i, j]
    
    return scaled_room


def generate_scaled_grid(path, scale_factor):
    """
    Vectorized generation of a 3x3 grid for each path point.

    Args:
        path: Original path, shape (N, 2)
        scale_factor: Scale factor

    Returns:
        3x3 grids for each path point, shape (N, 9, 2)
    """
    step = scale_factor / 2  # Spacing between grid points

    # Compute lower-left corner for each path point
    lower_left = path * scale_factor

    # Grid offsets
    offsets = np.array([
        [0, 0], [step, 0], [2*step, 0],
        [0, step], [step, step], [2*step, step],
        [0, 2*step], [step, 2*step], [2*step, 2*step]
    ])

    # Broadcast path points to match offset shape
    grids = lower_left[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    
    return grids

def scale_pos(scaled_room, robot_pos, goal_pos, min_distance=2, scale_factor=None):
    while True:
        # Pick final positions within the scaled room
        final_robot_pos = [robot_pos[0] * 10 + np.random.randint(0, 10), robot_pos[1] * 10 + np.random.randint(0, 10)]
        final_goal_pos = [goal_pos[0] * 10 + np.random.randint(0, 10), goal_pos[1] * 10 + np.random.randint(0, 10)]
        if scaled_room[tuple(final_robot_pos)] == 0 and scaled_room[tuple(final_goal_pos)] == 0:
            if is_far_from_obstacles(scaled_room, final_robot_pos, min_distance) and is_far_from_obstacles(scaled_room, final_goal_pos, min_distance):
                scaled_room[tuple(final_robot_pos)] = -2e-3
                scaled_room[tuple(final_goal_pos)] = -3e-3
                return scaled_room
            
def unscale_room(scaled_room, scale_factor=5):
    """Downscale a room back to its original size."""
    original_size = scaled_room.shape[0] // scale_factor
    unscaled_room = np.zeros((original_size, original_size), dtype=float)

    for i in range(original_size):
        for j in range(original_size):
            # Get corresponding block in the scaled room
            block = scaled_room[i * scale_factor:(i + 1) * scale_factor, j * scale_factor:(j + 1) * scale_factor]
            # Use the block center value
            unscaled_room[i, j] = block[scale_factor // 2, scale_factor // 2]

    return unscaled_room

def create_rand_room(level, grid_size=10, target_size=100, min_distance=2, set_pos=False):
    """Generate a valid terrain with navigable paths."""
    room = create_room(grid_size) # grid_size = 20, room_size = 20*20
    room = add_obstacles(room, level, grid_size) # obstacle height 0.2m ~ 1.0m
    scale_factor = target_size / grid_size # target_size = 100, scale_factor = 5
    scaled_room = scale_room(room, scale_factor=scale_factor) # scaled_room_size = 100*100
    if set_pos:
        robot_pos, goal_pos = place_robot_and_goal(room)
        final_robot_pos, final_goal_pos = scale_robot_and_goal(robot_pos, goal_pos, scale_factor=scale_factor)
        scaled_room[tuple(final_robot_pos)] = -2e-3
        scaled_room[tuple(final_goal_pos)] = -3e-3

    return scaled_room


def visualize_room_goal_and_robot(room):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Create a simplified map for visualization
    # 0: Ground, 1: Obstacles (any value > 0)
    viz_map = np.zeros_like(room)
    viz_map[room > 0.01] = 1 
    
    # Custom colors: 0 -> White (Ground), 1 -> Gray (Obstacles)
    cmap = ListedColormap(['#FFFFFF', '#404040'])
    
    plt.figure(figsize=(10, 10))
    # origin='lower' to match coordinate systems where (0,0) is bottom-left
    plt.imshow(viz_map, cmap=cmap, origin='lower')
    
    # Highlight robot and goal positions (extracting special values)
    # Robot = -2e-3, Goal = -3e-3
    robot_indices = np.argwhere(np.abs(room - (-2e-3)) < 1e-5)
    goal_indices = np.argwhere(np.abs(room - (-3e-3)) < 1e-5)
    
    # Note: robot_indices stores [row, col]. In scatter(x, y), row is Y, col is X.
    if robot_indices.size > 0:
        plt.scatter(robot_indices[:, 1], robot_indices[:, 0], c='#3498db', label='Robot', marker='o', s=150, edgecolors='black', linewidth=2)
    if goal_indices.size > 0:
        plt.scatter(goal_indices[:, 1], goal_indices[:, 0], c='#e74c3c', label='Goal', marker='X', s=200, edgecolors='black', linewidth=2)
    
    plt.title('Terrain Visualization: Ground (White), Obstacles (Gray)', fontsize=14)
    plt.xlabel('X Grid (m)', fontsize=12)
    plt.ylabel('Y Grid (m)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.grid(True, which='both', color='lightgrey', linestyle='--', alpha=0.5)
    plt.show()
   
if __name__ == "__main__":
    # Create terrains for different levels
    level = 9
    grid_size = 20
    target_size = 100
    for i in range(100):
        scaled_room = create_rand_room(level, grid_size, target_size=target_size, set_pos=False)
        robot_pos, goal_pos = place_robot_and_goal(scaled_room)
        scaled_room[tuple(robot_pos)] = -2e-3
        scaled_room[tuple(goal_pos)] = -3e-3
        visualize_room_goal_and_robot(scaled_room)





