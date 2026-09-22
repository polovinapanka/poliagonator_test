from stable_baselines3 import PPO

from UE_env import UnrealEnv


MODEL_PATH = "ppo_unreal_100000steps"

NUM_EPISODES = 10


def main():
    env = UnrealEnv()

    model = PPO.load(MODEL_PATH)

    successes = 0
    truncated = 0

    total_steps = 0
    total_rewards = 0.0

    print("=" * 60)
    print("ТЕСТ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 60)

    for episode in range(1, NUM_EPISODES + 1):

        observation, info = env.reset()

        episode_reward = 0.0
        episode_steps = 0

        print(f"\n--- Эпизод {episode}/{NUM_EPISODES} ---")

        print(
            f"Start: "
            f"Agent=({observation[0]:.0f}, {observation[1]:.0f}) "
            f"Target=({observation[2]:.0f}, {observation[3]:.0f})"
        )

        terminated = False
        episode_truncated = False

        while not terminated and not episode_truncated:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = int(action)

            observation, reward, terminated, episode_truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            print(
                f"Step {episode_steps:3d}: "
                f"Action={action} | "
                f"Agent=({observation[0]:.0f}, {observation[1]:.0f}) | "
                f"Distance={info['distance']:.1f} | "
                f"Reward={reward:.2f}"
            )

        total_steps += episode_steps
        total_rewards += episode_reward

        if terminated:
            successes += 1

            print(
                f"SUCCESS! "
                f"Цель достигнута за {episode_steps} шагов."
            )

        elif episode_truncated:
            truncated += 1

            print(
                f"TRUNCATED! "
                f"Лимит {episode_steps} шагов."
            )

        print(
            f"Итог эпизода: "
            f"Reward={episode_reward:.2f}, "
            f"Steps={episode_steps}, "
            f"Distance={info['distance']:.1f}"
        )

    env.close()

    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)

    success_rate = successes / NUM_EPISODES * 100
    average_steps = total_steps / NUM_EPISODES
    average_reward = total_rewards / NUM_EPISODES

    print(f"Эпизодов:             {NUM_EPISODES}")
    print(f"Успешных:             {successes}")
    print(f"Неуспешных:           {truncated}")
    print(f"Success rate:         {success_rate:.1f}%")
    print(f"Среднее число шагов:  {average_steps:.1f}")
    print(f"Средняя награда:      {average_reward:.2f}")

    print("\n" + "=" * 60)

    if successes == NUM_EPISODES:
        print("Модель успешно прошла все 10 эпизодов.")

    elif successes >= NUM_EPISODES * 0.7:
        print("Модель успешно решает большую часть эпизодов.")

    elif successes > 0:
        print("Модель иногда достигает цели, но обучение ещё нестабильно.")

    else:
        print("Модель не смогла достичь цели ни в одном эпизоде.")

    print("=" * 60)


if __name__ == "__main__":
    main()
