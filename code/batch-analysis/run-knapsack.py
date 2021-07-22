import warnings

import json

from knapsack_utils import *
import pandas as pd
import os

from deap import base, creator
from deap.benchmarks.tools import hypervolume
import deap.tools as tools
from deap.tools.emo import *

warnings.filterwarnings("ignore")


def run_one_alg(algo, knapsack_seed, pop_seed, n_k, freq_cald_hypervolume, mating_algo, sel_for_mating, n_i, pop_size,
                ngen, file_name, final_front_file_name):
    start = time.time()
    pop, stats, pop_arr = main(algo=algo, knapsack_seed=knapsack_seed, pop_seed=pop_seed, n_i=n_i, n_k=n_k,
                               pop_size=pop_size, NGEN=ngen, knapsack=None,
                               is_unique_solutions=True, num_iter_repair=150, mating_algo=mating_algo,
                               sel_for_mating=sel_for_mating,
                               freq_cald_hypervolume=freq_cald_hypervolume, is_save_all_gen=False, )
    end = time.time()
    # save results to a data-frame and then to file
    v = {k: [dic[k] for dic in stats] for k in stats[0]}
    stats_df = pd.DataFrame(v)
    stats_df.set_index('gen', inplace=True)
    stats_df.to_csv(file_name)
    # return elapsed time

    if final_front_file_name is not None:
        front_1 = deap.tools.emo.sortNondominated(pop, len(pop), first_front_only=True)[0]
        last_pop_fit = [list(ind.fitness.values) for ind in front_1]
        with open(final_front_file_name, 'w') as outfile:
            json.dump(last_pop_fit, outfile)
        pass
    return end - start


def run():
    n_k_arr = [
        #2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        15,
        25]
    n_runs = 30
    algo_arr = [
        #'WS',
        #'AR'
        'NSGA-2',
        # 'NSGA-3',
        # 'PO-count',
        #'PO-prob',
        # 'PO-prob-repair',
    ]

    sel_for_mating_arr = [
        #только рандом и турнамент для теста
        'random',
        'tournament',
        # 'best',
        # 'uniform',
    ]
    mating_algo_arr = [
        #только униформ
        'uniform',
        # 'cxOnePoint'
    ]

    pop_size = 250
    ngen = 500
    n_i = 250

    freq_cald_hypervolume_arr = [##1000,
                                 #None,
                                 1000,
                                 1000,
                                 1000,
                                 1000,
                                 1000,
                                 5000,
                                 5000,
                                 5000,
                                 5000,
                                 ]

    tot_runs = len(algo_arr) * len(sel_for_mating_arr) * len(mating_algo_arr) * n_runs * len(n_k_arr)
    pos = 0
    print('Start, tot = {}'.format(tot_runs))
    counter = -1

    for n_k in n_k_arr:
        counter += 1
        path = '../results/n_k-{}_new/'.format(n_k)
        # create output dir if required
        if not os.path.exists(path):
            os.makedirs(path)

        for algo in algo_arr:
            for sel_for_mating in sel_for_mating_arr:
                for mating_algo in mating_algo_arr:
                    for i in range(0, n_runs):
                        # seed_val_arr = [np.random.randint(0, 100)]
                        # for i in seed_val_arr:
                        # for i in range(4, 5):
                        file_name = path + '{}_mat_sel_{}_mat_{}_seed_{}.csv'.format(algo, sel_for_mating, mating_algo,
                                                                                     i)
                        final_front_file_name = path + '{}_mat_sel_{}_mat_{}_seed_{}_final_front.csv'.format(algo,
                                                                                                             sel_for_mating,
                                                                                                             mating_algo,
                                                                                                             i)
                        # frequency to calculate hypervolume
                        # take from array
                        #if n_k < len(freq_cald_hypervolume_arr):
                        #    freq_cald_hypervolume = freq_cald_hypervolume_arr[n_k]
                        #else:
                        #    freq_cald_hypervolume = freq_cald_hypervolume_arr[-1]
                        #    pass

                        # for statistical results set None

                        # the same value for all
                        freq_cald_hypervolume = 10 #freq_cald_hypervolume_arr[counter]


                        tot_time = run_one_alg(algo=algo, knapsack_seed=i, pop_seed=i, n_k=n_k,
                                               freq_cald_hypervolume=freq_cald_hypervolume, mating_algo=mating_algo,
                                               sel_for_mating=sel_for_mating, n_i=n_i, pop_size=pop_size,
                                               ngen=ngen, file_name=file_name,
                                               final_front_file_name=final_front_file_name)
                        pos += 1
                        print('{} / {}: {} - {}'.format(pos, tot_runs, file_name, tot_time))
                        pass
                    pass
                pass
            pass
        pass

    pass


def run_time_func_pop_size():
    # n_k_arr = [2, 5, 10, 15, 25]
    n_k_arr = [10]

    algo_arr = [
        'NSGA-2',
        # 'PO-prob',
        # 'NSGA-3',
        # 'PO-count',
        # 'PO-prob-repair',
    ]

    sel_for_mating_arr = [
        'random',
    ]
    mating_algo_arr = [
        'uniform',
        # 'cxOnePoint'
    ]

    pop_size_arr = [i * 50 for i in range(1, 11)]
    ngen = 101
    n_i = 250
    freq_cald_hypervolume = ngen + 1

    tot_runs = len(algo_arr) * len(sel_for_mating_arr) * len(mating_algo_arr) * len(pop_size_arr) * len(n_k_arr)
    pos = 0
    print('Start, tot = {}'.format(tot_runs))

    path = 'results_pop_size/'
    # create output dir if required
    if not os.path.exists(path):
        os.makedirs(path)

    seed_val = 0
    for n_k in n_k_arr:
        for algo in algo_arr:
            for sel_for_mating in sel_for_mating_arr:
                for mating_algo in mating_algo_arr:
                    for pop_size in pop_size_arr:
                        file_name = path + 'nk_{}_{}_mat_sel_{}_mat_{}_seed_{}_pop_size_{}.csv'.format(n_k, algo,
                                                                                                       sel_for_mating,
                                                                                                       mating_algo,
                                                                                                       seed_val,
                                                                                                       pop_size)
                        tot_time = run_one_alg(algo=algo, knapsack_seed=seed_val, pop_seed=seed_val, n_k=n_k,
                                               freq_cald_hypervolume=freq_cald_hypervolume, mating_algo=mating_algo,
                                               sel_for_mating=sel_for_mating, n_i=n_i, pop_size=pop_size,
                                               ngen=ngen, file_name=file_name,
                                               final_front_file_name=None)
                        pos += 1
                        print('{} / {}: {} - {}'.format(pos, tot_runs, file_name, tot_time))
                        pass
                    pass
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    print('Hello')
    # main()
    run()
    # run_time_func_pop_size()
    pass
