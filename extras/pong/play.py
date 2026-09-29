import pygame
from pong_env import ToyPongEnv

env = ToyPongEnv(render_mode="human", representation="full")
obs, info = env.reset()

print("Initial observation:", obs)
print("Controls: W/Up, S/Down, R restart, Esc quit")

running = True
while running:
    action = 1  # stay

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                obs, info = env.reset()
                print("\n--- RESET: new episode ---")
                print("Initial observation:", obs)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        action = 0
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        action = 2

    old_obs = obs.copy()
    obs, reward, terminated, truncated, info = env.step(action)

    if reward != 0:
        print(
            f"obs={old_obs.round(2)} action={action} "
            f"reward={reward:+.2f} -> new_obs={obs.round(2)} "
            f"score={info['player_score']}-{info['opponent_score']}"
        )

    if terminated or truncated:
        print("Episode ended:", info)
        obs, info = env.reset()

env.close()
