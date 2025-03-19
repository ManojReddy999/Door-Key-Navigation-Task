from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

def motion_model_A(env,state,action):
    x,y,orientation,k,d = state
    
    move = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    dx, dy = move[orientation]            # Calculating dx and dy based on the current orientation

    if 0 <= x+dx < env.grid.width and 0 <= y+dy < env.grid.height:                             # checking for boundaries
        if action == MF:                                                                       # if there is a wall or there is a locked door, the agent can not move forward
            if env.grid.get(x+dx,y+dy) is not None and env.grid.get(x+dx,y+dy).type == 'wall':
                return state
            elif env.grid.get(x+dx,y+dy) is not None and env.grid.get(x+dx, y+dy).type == 'door' and d == 0:
                return state
            else:
                return (x+dx, y+dy, orientation,k,d)
        
        elif action in [TL, TR]:                                      
            if action == TL:
                new_orientation = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}         # Turn left or right is implemented based on these dictionaries
            else:
                new_orientation = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
            return (x, y, new_orientation[orientation],k,d)
        
        elif action == PK and env.grid.get(x+dx,y+dy) is not None and env.grid.get(x+dx, y+dy).type == 'key' and k == 0:   # if the agent doesn't have the key and there is a key in front
            return (x,y,orientation,1,d)
        elif action == UD and env.grid.get(x+dx,y+dy) is not None and env.grid.get(x+dx, y+dy).type == 'door' and k == 1 and d == 0: # if the agent has a key and there is a locked door in front
            return (x,y,orientation,k,1)
        else:
            return (x, y, orientation,k,d)    
    else:
        return (x, y, orientation,k,d)
    

def doorkey_problem(env,info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    """
    height = env.unwrapped.height
    width = env.unwrapped.width

    orientation = ['E','W','N','S']

    X = [(x,y,o,k,d) for x in range(1,height-1) for y in range(1,width-1) for o in orientation for k in range(2) for d in range(2)]  # State Space
    U = {MF,TR,TL,PK,UD}                                                                                                             # Action Space

    VT = {state: np.inf for state in X}
    for state in X:
        if np.all((state[0], state[1]) == info['goal_pos']):                        # Terminal Cost - zero at goal states, infinity otherwise
            VT[state] = 0

    # stage_cost = 1
    stage_cost = {state: 1 for state in X}
    for state in X:
        if np.all((state[0], state[1]) == info['goal_pos']):                        # Stage cost - 1 for all states except goal states and action, 0 for terminal states
            stage_cost[state] = 0

    policy = {}
    for t in range(20,-1,-1):                                                                                # Horizon is set to 20
        for state in X:
            Q = {}
            for action in U:
                Q[action] = stage_cost[state] + VT[motion_model_A(env,state,action)]     # Value at each state for each action is calculated and minimized
            min_action = min(Q,key = Q.get)                                         
            VT[state] = min(Q.values())
            policy[state] = min_action                                                   # Action corresponding to the minimum value is stored
    
    initial_position = info['init_agent_pos']                                            # Initial position
    init_orientation = info['init_agent_dir']                                            # Initial direction
    door = env.grid.get(info["door_pos"][0], info["door_pos"][1])                        # Door position
    orient = {(0, -1) : 'N', (1, 0) : 'E', (0, 1) : 'S', (-1, 0) : 'W'}     
    if env.carrying is None:
        k = 0                                                                            # Key status
    else:
        k = 1
    
    if door.is_open:
        d = 1                                                                            # Door status
    else:
        d = 0

    current_state = (initial_position[0],initial_position[1],orient[(init_orientation[0],init_orientation[1])],k,d)  # Initial State
    path = [current_state]
    optim_act_seq = []

    while np.any((current_state[0], current_state[1]) != info['goal_pos']):     # Extracting the control sequence
        action = policy[current_state]
        optim_act_seq.append(action)
        current_state = motion_model_A(env, current_state, action)
        path.append(current_state)

    return optim_act_seq



def partA():
    env_path = "envs/known_envs/example-8x8.env"
    env, info = load_env(env_path)  # load an environment
    seq = doorkey_problem(env,info)  # find the optimal action sequence
    name = env_path[len(".envs/known_envs/doorkey-"):-len(".env")]
    draw_gif_from_seq(seq,env,"./gif/Known_gifs/"+name+".gif")

def motion_model_B(state,action,doors = [(4,2),(4,5)],keys = [(1, 1), (2, 3), (1, 6)],wall = [(4,0),(4,1),(4,3),(4,4),(4,6),(4,7)]):
    x,y,orientation,k,d0,d1,ki,gi = state
    # x,y,k,d = int(x),int(y),int(k),int(d)                                         # Motion model for Random map case
    
    move = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
    dx, dy = move[orientation]
    # new_x,new_y = env.front_pos[0],env.front_pos[1]

    if 0 <= x+dx < 8 and 0 <= y+dy < 8:                                             # Checking for boundaries
        if action == MF:                                                            # Can only move forward if the cell is not a wall or if there is an open door
            if (x+dx,y+dy) in wall:
                return state
            elif ((x+dx,y+dy) == doors[0] and d0 == 0) or ((x+dx,y+dy) == doors[1] and d1 == 0):
                return state
            else:
                return (x+dx,y+dy,orientation,k,d0,d1,ki,gi) 
        
        elif action in [TL, TR]:
            if action == TL:
                new_orientation = {'N': 'W', 'W': 'S', 'S': 'E', 'E': 'N'}
            else:
                new_orientation = {'N': 'E', 'E': 'S', 'S': 'W', 'W': 'N'}
            return (x, y, new_orientation[orientation],k,d0,d1,ki,gi)
        
        elif action == PK and k == 0 and (x+dx,y+dy) == keys[ki]:                   # Can pickup the key if the next cell has a key and the agent does not have the key
            return (x,y,orientation,1,d0,d1,ki,gi)
        elif action == UD and k == 1:                                               # Can unlock the door if there is a locked door and the agent has the key
            if (d0 == 0 and (x+dx,y+dy) == doors[0]):
                return (x,y,orientation,k,1,d1,ki,gi)
            elif (d1 == 0 and (x+dx,y+dy) == doors[1]):
                return (x,y,orientation,k,d0,1,ki,gi)
            else:
                return state
        else:
            return state   
    else:
        return state

def compute_policy(goals = [(5, 1), (6, 3), (5, 6)]):                              # Function to compute the policy for the random map case
    orientation = ['E','W','N','S']
    X = [(x,y,o,k,d0,d1,ki,gi) for x in range(0,8) for y in range(0,8) for o in orientation for k in range(2) for d0 in range(2) for d1 in range(2) for ki in range(3) for gi in range(3)]
    U = {MF,TR,TL,PK,UD}                                                           # State Space and Action Space

    VT = {state: np.inf for state in X}
    for state in X:
        if np.all((state[0], state[1]) == goals[state[-1]]):                       # Initializing the terminal cost
            VT[state] = 0

    stage_cost = {state: 1 for state in X}
    for state in X:
        if np.all((state[0], state[1]) == goals[state[-1]]):                       # Initializing the Stage cost
            stage_cost[state] = 0

    policy_B = {state:None for state in X}                                          
    for t in range(30,-1,-1):                                                     # Computing the policy 
        for state in X:
            Q = {}
            for action in U:
                Q[action] = stage_cost[state] + VT[motion_model_B(state,action)]   # Updating the value function and minimising it
            min_action = min(Q,key = Q.get)
            VT[state] = min(Q.values())
            policy_B[state] = min_action                                           # Storing the action that minimizes the value

    return policy_B

Policy_Random_Maps = compute_policy()                                              # Computing the policy

def Extract_actions(info,policy=Policy_Random_Maps,keys = [(1, 1), (2, 3), (1, 6)],goals = [(5, 1), (6, 3), (5, 6)]):
                                                                                   # Function to compute the action sequence from  the policy
    initial_position = info['init_agent_pos']
    init_orientation = info['init_agent_dir']
    orient = {(0, -1) : 'N', (1, 0) : 'E', (0, 1) : 'S', (-1, 0) : 'W'}            # Extracting the initial state of the agent

    if info['door_open'][0]:
        d0 = 1                                                                     # Extracting the initial status of door 1
    else:
        d0 = 0

    if info['door_open'][1]:
        d1 = 1                                                                     # Extracting the initial status of door 1
    else:
        d1 = 0

    for i in range(3):
        if np.all(info['key_pos'] == keys[i]):                
            ki = i                                                                 # Extracting the key position
    
    for j in range(3):
        if np.all(info['goal_pos'] == goals[j]):                                   # Extracting the goal positions
            gi = j
    
    current_state = (initial_position[0],initial_position[1],orient[(init_orientation[0],init_orientation[1])],0,d0,d1,ki,gi)    # Initializing the state
    path = [current_state]
    actions = []

    while np.any((current_state[0], current_state[1]) != info['goal_pos']):       # Extracting the action sequence form the policy
        action = policy[current_state]
        actions.append(action)
        current_state = motion_model_B(current_state, action)
        path.append(current_state)

    return actions

def partB():
    env_folder = "./envs/random_envs"
    env, info, env_path = load_random_env(env_folder)                             # Load a random environment
    sequence = Extract_actions(info)                                              # Extract the action sequence from  the policy
    name = env_path[len(".envs/random_envs/doorkey-"):-len(".env")]
    draw_gif_from_seq(sequence,env,"./gif/random_gifs/"+name+".gif")



if __name__ == "__main__":
    # example_use_of_gym_env()
    partA()
    # partB()








# action_map = {
#     0: 'MF',  # Move Forward
#     1: 'TL',  # Turn Left
#     2: 'TR',  # Turn Right
#     3: 'PK',  # Pickup Key
#     4: 'UD'   # Unlock Door                                                 # Code to run dynamic programming and storing the control sequences for all known maps
# }

# env_list = [os.path.join('envs/known_envs', env_file) for env_file in os.listdir('envs/known_envs') if env_file.endswith('.env')]
# sequences = []
# for i in env_list:
#     env,info = load_env(i)
#     sequence = doorkey_problem(env,info)
#     name = i[len("envs/known_envs/doorkey-"):-len(".env")]
#     draw_gif_from_seq(sequence,env,"./gif/"+name+".gif")
#     sequence = [action_map[num] for num in sequence]
#     sequences.append(sequence)
#     print(sequence)

# from PIL import Image

# def extract_frames_from_gif(gif_path, output_folder):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder, exist_ok=True)
    
#     with Image.open(gif_path) as img:
#         # Check if the image is animated (i.e., has multiple frames)         # Code to extract images form a GIF
#         if not img.is_animated:
#             print(f"The image is not animated.")
#             return

#         frame_count = img.n_frames
#         for i in range(frame_count):
#             img.seek(i)  # Move to the frame index i
#             img.save(f'{output_folder}/frame_{i:03d}.png')  # Save frame as PNG

# Example usage
# gif_path = 'gif/8x8-direct.gif'
# output_folder = '/home/mmkr/Documents/ECE276B_PR1/starter_code/Images/8x8-direct'
# extract_frames_from_gif(gif_path, output_folder)