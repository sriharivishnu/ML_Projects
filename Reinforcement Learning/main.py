import gym
from QNetwork import DeepQNetwork, Agent
import numpy as np
from gym import wrappers

def preprocess(observation):
    return np.mean(observation[30:, :], axis=2).reshape(180, 160, 1)

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames [0:buffer_size-1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size-1, :] = frame

    stacked_frames = stacked_frames.reshape(1, *frame.shape[0:2], buffer_size)

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.00025, input_dims=(180, 160, 4), n_actions=3, mem_size=20000, batch_size=32)
    if load_checkpoint:
        agent.load_models()
    scores = []
    numGames = 200
    stack_size = 4
    score = 0

    while agent.mem_cntr < 20000:
        done = False
        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = np.random.choice([0,1,2])
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
            action -= 1
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_

    print('Done with random gameplay, game on.')

    for i in range(numGames):
        done = False
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[max(0,i-10):(i+10)])
            print ('episode', i, 'score', score, 'average_score %.3f' % avg_score, 'epsilon %.3f' % agent.epsilon)
<<<<<<< HEAD
            agent.save_models()
=======
>>>>>>> b986bdc8e2a50ad9f75796f7ba39fab23eb64cc5

        else:
            print ('episode: ', i, 'score', score)
            agent.save_models()

        observation = env.reset()
        observation = preprocess(observation)
        stacked_frames = None
        observation = stack_frames(stacked_frames, observation, stack_size)
        while not done:
            action = np.random.choice([0,1,2])
            action += 1
            observation_, reward, done, info = env.step(action)
            observation_ = stack_frames(stacked_frames, preprocess(observation_), stack_size)
            action -= 1
            agent.store_transition(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()
            score += reward
            env.render()
    scores.append(score)
