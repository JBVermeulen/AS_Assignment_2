from model import BangladeshModel
import pandas as pd

from model import BangladeshModel

SINGLE_RUN = True # Set to False to run the scenario experiment

# ---------------------------------------------------------------
def single_run():
    """Run a single scenario with a given seed and print output at terminal"""
    # run time 5 x 24 hours; 1 tick 1 minute
    run_length = 5 * 24 * 60

    # run time 1000 ticks
    # run_length = 1000

    seed = 1234567

    scenarios = {
        0: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 0},
        1: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 5},
        2: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 10},
        3: {'CatA': 0, 'CatB': 0, 'CatC': 5, 'CatD': 10},
        4: {'CatA': 0, 'CatB': 0, 'CatC': 10, 'CatD': 20},
        5: {'CatA': 0, 'CatB': 5, 'CatC': 10, 'CatD': 20},
        6: {'CatA': 0, 'CatB': 10, 'CatC': 20, 'CatD': 40},
        7: {'CatA': 5, 'CatB': 10, 'CatC': 20, 'CatD': 40},
        8: {'CatA': 10, 'CatB': 20, 'CatC': 40, 'CatD': 80}
    }

    sim_model = BangladeshModel(seed=seed, scenario=scenarios[4])

    # Check if the seed is set
    print("SEED " + str(sim_model._seed))

    # One run with given steps
    for i in range(run_length):
        sim_model.step()

    df = pd.DataFrame(sim_model.wait_events)
    df.to_csv('../experiment/scenario_none.csv')

def scenario_experiment():
    """Run multiple scenarios with different seeds and print output at terminal"""
    run_length = 5 * 24 * 60

    scenarios = {
        0: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 0},
        1: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 5},
        2: {'CatA': 0, 'CatB': 0, 'CatC': 0, 'CatD': 10},
        3: {'CatA': 0, 'CatB': 0, 'CatC': 5, 'CatD': 10},
        4: {'CatA': 0, 'CatB': 0, 'CatC': 10, 'CatD': 20},
        5: {'CatA': 0, 'CatB': 5, 'CatC': 10, 'CatD': 20},
        6: {'CatA': 0, 'CatB': 10, 'CatC': 20, 'CatD': 40},
        7: {'CatA': 5, 'CatB': 10, 'CatC': 20, 'CatD': 40},
        8: {'CatA': 10, 'CatB': 20, 'CatC': 40, 'CatD': 80}
    }

    for key, value in scenarios.items():
        list_of_runs_wait_events = []
        list_of_runs_travel_time = []
        for seed_var in range(1, 11):
            seed = 123 + seed_var  # Different seed for each scenario
            print(f"Running scenario {key} with seed {seed}.")
            sim_model = BangladeshModel(seed=seed, scenario=value)
            list_of_runs_wait_events = []
            for seed_var in range(1, 11):
                seed = 123 + seed_var  # Different seed for each scenario
                print(f"Running scenario {key} with seed {seed}.")
                sim_model = BangladeshModel(seed=seed, scenario=value)

        for i in range(run_length):
            sim_model.step()

        df_wait_events = pd.DataFrame(sim_model.wait_events)
        df_wait_events['seed'] = seed
        list_of_runs_wait_events.append(df_wait_events)

    full_df_wait_events = pd.concat(list_of_runs_wait_events,ignore_index=True)
    full_df_wait_events.to_csv(f'../experiment/scenario{key}.csv')

if __name__ == "__main__":
    if SINGLE_RUN:
        single_run()
    else:
        scenario_experiment()

