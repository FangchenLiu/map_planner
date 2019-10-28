from gym.envs.registration import register
import gym

robots = ['Point', 'Ant']
task_types = ['Maze', 'Maze1', 'Push', 'Fall', 'Block', 'BlockMaze']
all_name = [x + y for x in robots for y in task_types]
for name_t in all_name:
    for Test in ['', 'Test']:
        #print(name_t)
        max_timestep = 200
        random_start = True
        if name_t[-4:] == 'Maze' or name_t[-4:] == 'aze1':
            goal_args = [[-4, -4], [20, 20]]
        if Test == 'Test':
            goal_args = [[0.0, 16.0], [1e-3,16 + 1e-3]]
            random_start = False
            max_timestep = 300

        register(
            id=name_t + Test + '-v0',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 8, 'max_timestep':max_timestep, 'random_start':random_start},
        )

        register(
            id=name_t + Test + '-v1',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 4, 'max_timestep':max_timestep, 'random_start':random_start},
        )

        register(
            id=name_t + Test + '-v2',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 2, 'max_timestep':max_timestep, 'random_start':random_start},
        )

        register(
            id=name_t + Test + '-v3',
            entry_point='goal_env.mujoco.create_maze_env:create_maze_env',
            kwargs={'env_name': name_t, 'goal_args': goal_args, 'maze_size_scaling': 6, 'max_timestep':max_timestep, 'random_start':random_start},
        )


register(
    id='PointMazeX-v0',
    entry_point='goal_env.mujoco.point_maze_env:PointMazeEnv',
    kwargs={
        "maze_size_scaling": 4,
        "maze_id": "Maze1",
        "maze_height": 0.5,
        "manual_collision": True,
        "goal": (1, 3),
    }
)

register(
    id='PointMazeX-v1',
    entry_point='goal_env.mujoco.point_maze_env:PointMazeEnv',
    kwargs={
        "maze_size_scaling": 2,
        "maze_id": "Maze1",
        "maze_height": 0.5,
        "manual_collision": True,
        "goal": (1, 3),
    }
)


register(
    id='AntMazeX-v0',
    entry_point='goal_env.mujoco.ant_maze_env:AntMazeEnv',
    kwargs={
        "maze_size_scaling": 8,
        "maze_id": "Maze",
        "maze_height": 0.5,
    }
)


register(
    id='AntMazeX-v1',
    entry_point='goal_env.mujoco.ant_maze_env:AntMazeEnv',
    kwargs={
        "maze_size_scaling": 4,
        "maze_id": "Maze",
        "maze_height": 0.5,
        "goal": (1, 3),
    }
)