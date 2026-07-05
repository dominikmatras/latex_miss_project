import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LICZBA_AGENTOW = 100
LICZBA_ITERACJI = 100
MAJATEK_POCZATKOWY = 100

MIN_DOCHOD = 5
MAX_DOCHOD = 15

MIN_KOSZT = 3
MAX_KOSZT = 10

LICZBA_TRANSAKCJI_NA_ITERACJE = 50

ZIARNO_LOSOWOSCI = 10

class Agent:
    def __init__(self, agent_id, wealth):
        self.agent_id = agent_id
        self.wealth = wealth

    def receive_income_and_pay_costs(self):
        income = random.randint(MIN_DOCHOD, MAX_DOCHOD)
        expenses = random.randint(MIN_KOSZT, MAX_KOSZT)

        self.wealth += income
        self.wealth -= expenses

        # Majątek nie może spaść poniżej zera
        if self.wealth < 0:
            self.wealth = 0

def calculate_gini(values):
    values = np.array(values, dtype=float)
    n = len(values)

    if np.mean(values) == 0:
        return 0

    differences_sum = np.abs(values[:, None] - values[None, :]).sum()
    gini = differences_sum / (2 * n * n * np.mean(values))

    return gini

def run_simulation(redistribution_rate=0.0):
    """
    redistribution_rate:
    0.00 oznacza brak redystrybucji,
    0.05 oznacza redystrybucję 5%,
    0.10 oznacza redystrybucję 10%.
    """

    random.seed(ZIARNO_LOSOWOSCI)

    agents = [
        Agent(agent_id=i, wealth=MAJATEK_POCZATKOWY)
        for i in range(LICZBA_AGENTOW)
    ]

    history = []

    for step in range(LICZBA_ITERACJI + 1):

        wealth_values = [agent.wealth for agent in agents]

        history.append({
            "iteracja": step,
            "sredni_majatek": np.mean(wealth_values),
            "minimalny_majatek": np.min(wealth_values),
            "maksymalny_majatek": np.max(wealth_values),
            "odchylenie_standardowe": np.std(wealth_values),
            "gini": calculate_gini(wealth_values),
            "calkowity_majatek": np.sum(wealth_values)
        })

        if step == LICZBA_ITERACJI:
            break

        # 1. Dochody i koszty życia
        for agent in agents:
            agent.receive_income_and_pay_costs()

        # 2. Losowe transakcje między agentami
        for _ in range(LICZBA_TRANSAKCJI_NA_ITERACJE):
            agent_a, agent_b = random.sample(agents, 2)

            amount = random.randint(1, 10)

            if random.random() < 0.5:
                payer = agent_a
                receiver = agent_b
            else:
                payer = agent_b
                receiver = agent_a

            transfer = min(amount, payer.wealth)

            payer.wealth -= transfer
            receiver.wealth += transfer

        # 3. Redystrybucja majątku
        if redistribution_rate > 0:
            taxes = [agent.wealth * redistribution_rate for agent in agents]
            tax_pool = sum(taxes)
            redistribution_amount = tax_pool / LICZBA_AGENTOW

            for agent, tax in zip(agents, taxes):
                agent.wealth = agent.wealth - tax + redistribution_amount

    final_wealth = [agent.wealth for agent in agents]
    history_df = pd.DataFrame(history)

    return history_df, final_wealth



def main():
    os.makedirs("figures", exist_ok=True)

    # Trzy warianty eksperymentu
    df_0, wealth_0 = run_simulation(redistribution_rate=0.00)
    df_5, wealth_5 = run_simulation(redistribution_rate=0.05)
    df_10, wealth_10 = run_simulation(redistribution_rate=0.10)

    plt.figure(figsize=(10, 6))
    plt.plot(df_0["iteracja"], df_0["sredni_majatek"], label="średni majątek")
    plt.plot(df_0["iteracja"], df_0["minimalny_majatek"], label="minimalny majątek")
    plt.plot(df_0["iteracja"], df_0["maksymalny_majatek"], label="maksymalny majątek")
    plt.xlabel("Iteracja")
    plt.ylabel("Majątek")
    plt.title("Zmiana majątku agentów w czasie")
    plt.legend()
    plt.grid(True)
    plt.savefig("document/figures/wealth_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(df_0["iteracja"], df_0["gini"], label="redystrybucja 0%")
    plt.plot(df_5["iteracja"], df_5["gini"], label="redystrybucja 5%")
    plt.plot(df_10["iteracja"], df_10["gini"], label="redystrybucja 10%")
    plt.xlabel("Iteracja")
    plt.ylabel("Współczynnik Giniego")
    plt.title("Zmiana współczynnika Giniego w czasie")
    plt.legend()
    plt.grid(True)
    plt.savefig("document/figures/gini_over_time.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(wealth_0, bins=12)
    plt.xlabel("Majątek końcowy")
    plt.ylabel("Liczba agentów")
    plt.title("Końcowy rozkład majątku agentów")
    plt.grid(True)
    plt.savefig("document/figures/final_wealth_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    results = []

    for label, df in [
        ("0%", df_0),
        ("5%", df_5),
        ("10%", df_10)
    ]:
        last_row = df.iloc[-1]

        results.append({
            "Redystrybucja": label,
            "Średni majątek": round(last_row["sredni_majatek"], 2),
            "Minimum": round(last_row["minimalny_majatek"], 2),
            "Maksimum": round(last_row["maksymalny_majatek"], 2),
            "Odchylenie standardowe": round(last_row["odchylenie_standardowe"], 2),
            "Gini": round(last_row["gini"], 3)
        })

    results_df = pd.DataFrame(results)

    print("\nWyniki końcowe symulacji:\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("wyniki_symulacji.csv", index=False, encoding="utf-8-sig")

    print("\nWygenerowano pliki:")
    print("document/figures/wealth_over_time.png")
    print("document/figures/gini_over_time.png")
    print("document/figures/final_wealth_histogram.png")
    print("wyniki_symulacji.csv")


if __name__ == "__main__":
    main()