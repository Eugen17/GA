import numpy as np

import subprocess
import copy

import time

from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence
from deap import creator
import deap

from deap import base, creator
from deap.benchmarks.tools import hypervolume
import deap.tools as tools
from deap.tools.emo import *

from deap import tools, algorithms

weigh_key = 'W'
profit_key = 'P'
capacity_key = 'C'
q_key = 'Q'


def create_knapsack(n_i=7, n_k=5, low=10, high=100):
    w_ki = np.random.randint(low=low, high=high + 1, size=(n_k, n_i))
    p_ki = np.random.randint(low=low, high=high + 1, size=(n_k, n_i))
    c_k = 0.5 * w_ki.sum(axis=1)
    q_i = np.argsort((p_ki / w_ki).max(axis=0))
    return {weigh_key: w_ki, profit_key: p_ki, capacity_key: c_k, q_key: q_i}


def get_num_of_overw_k(knapsack, ind):
    return sum((knapsack[weigh_key] * ind).sum(axis=1) > knapsack[capacity_key])


def repair_individual_list(ind, knapsack):
    num_of_overw_k = get_num_of_overw_k(knapsack, ind)
    if num_of_overw_k == 0:
        return
    for pos in knapsack[q_key]:
        if ind[pos] != 0:
            ind[pos] = 0
            num_of_overw_k = get_num_of_overw_k(knapsack, ind)
            if num_of_overw_k == 0:
                break
        pass
    pass


def create_individual(n_i, knapsack):
    ind = np.random.randint(low=0, high=1 + 1, size=n_i)
    ind = ind.tolist()
    num_of_overw_k = get_num_of_overw_k(knapsack, ind)
    if num_of_overw_k > 0:
        repair_individual_list(ind, knapsack)
    return creator.Individual(ind)


def eval_knapsack(knapsack, ind):
    res = (knapsack[profit_key] * ind).sum(axis=1)
    return tuple(res)


def mate_f(knapsack, func, ind1, ind2):
    ind1, ind2 = func(ind1, ind2)
    repair_individual_list(ind1, knapsack)
    repair_individual_list(ind2, knapsack)
    return ind1, ind2


def mate_f_prob(knapsack, func, indpb, ind1, ind2):
    ind1, ind2 = func(ind1, ind2, indpb=indpb)
    repair_individual_list(ind1, knapsack)
    repair_individual_list(ind2, knapsack)
    return ind1, ind2


def mutate_f(knapsack, func, indpb, ind):
    ind = func(ind, indpb)[0]
    repair_individual_list(ind, knapsack)
    return ind,


def calc_hypervolume_java(front, num_test=10000000):
    start = time.time()
    fitness_vals_arr = [list(front[i].fitness.values) for i in range(0, len(front))]
    with open('fitness_vals.txt', 'w') as f:
        for line in fitness_vals_arr:
            line_str = ''
            for el in line:
                line_str += str(el) + ' '
                pass
            f.write(line_str[:-1] + '\n')
            pass
        pass
    subprocess.run(["java", "Hypervolume", str(num_test)])
    with open('hypervolume.txt') as f:
        for line in f:
            num_hits = int(line)
            break
    max_vals = np.array(fitness_vals_arr).max(axis=0)
    return num_hits, max_vals.prod() * num_hits / num_test, time.time() - start


def calc_hypervolume_java_arr(front, num_test=10000000):
    start = time.time()
    fitness_vals_arr = front
    with open('fitness_vals.txt', 'w') as f:
        for line in fitness_vals_arr:
            line_str = ''
            for el in line:
                line_str += str(el) + ' '
                pass
            f.write(line_str[:-1] + '\n')
            pass
        pass
    subprocess.run(["java", "Hypervolume", str(num_test)])
    with open('hypervolume.txt') as f:
        for line in f:
            num_hits = int(line)
            break
    max_vals = np.array(fitness_vals_arr).max(axis=0)
    return num_hits, max_vals.prod() * num_hits / num_test, time.time() - start


def get_num_dominated(pop_1, pop_2):
    front_1 = deap.tools.emo.sortNondominated(pop_1, len(pop_1), first_front_only=True)[0]
    front_2 = deap.tools.emo.sortNondominated(pop_2, len(pop_2), first_front_only=True)[0]
    # print(len(front_1))
    # print(len(front_2))
    return num_dominated(front_1, front_2)


def num_dominated(arr1, arr2):
    val_dominated = 0
    for el1 in arr1:
        for el2 in arr2:
            if sum(np.array(el2.fitness.values) > np.array(el1.fitness.values)) == len(el1.fitness.values):
                val_dominated += 1
                break
                pass
            pass
        pass
    return val_dominated, val_dominated / len(arr1), len(arr1)


class UpdateDist:
    def __init__(self, ax, all_pop_arr, legend_arr, n_obj, k_0=0, k_1=1):
        self.ref_point = np.zeros(n_obj)
        self.k_0 = k_0
        self.k_1 = k_1
        # print(len(self.ref_point))
        self.all_pop_arr = all_pop_arr
        # self.line, = ax.plot([], [])
        self.ax_pop = ax[0]
        self.line_arr = []
        for i in range(0, len(all_pop_arr)):
            self.line_arr.append(self.ax_pop.scatter(x=[], y=[], s=10))
        # self.line = self.ax_pop.scatter(x=[], y=[], s=10)
        self.ax_pop.legend(legend_arr)
        # Set up plot parameters
        self.ax_pop.set_xlim(4000, 11000)
        self.ax_pop.set_ylim(4000, 13000)
        self.ax_pop.grid(True)
        self.ax_pop.title.set_text('Evolution of generations')
        self.gen_text = self.ax_pop.text(0.05, 0.9, '', transform=self.ax_pop.transAxes)
        self.front_size_text = self.ax_pop.text(0.05, 0.8, '', transform=self.ax_pop.transAxes)

        self.ax_hypv = ax[1]
        # self.line_hypv, = self.ax_hypv.plot([], [])
        self.line_hypv_arr = []
        self.hypervolume_arr_list = []
        for i in range(0, len(all_pop_arr)):
            self.line_hypv_arr.append(self.ax_hypv.plot([], [])[0])
            self.hypervolume_arr_list.append([])

        self.ax_hypv.legend(legend_arr)
        self.ax_hypv.set_xlim(0, len(all_pop_arr[0]))
        min_hyp = calc_hypervolume_java(all_pop_arr[0][0], num_test=1000000)[
            1]  # (all_pop_arr[0][0], ref=self.ref_point)
        max_hyp = calc_hypervolume_java(all_pop_arr[0][-1], num_test=1000000)[
            1]  # (all_pop_arr[0][-1], ref=self.ref_point)
        for i in range(1, len(all_pop_arr)):
            tmp = calc_hypervolume_java(all_pop_arr[i][0], num_test=1000000)[
                1]  # (all_pop_arr[i][0], ref=self.ref_point)
            if tmp < min_hyp:
                min_hyp = tmp
                pass
            tmp = calc_hypervolume_java(all_pop_arr[i][-1], num_test=1000000)[
                1]  # (all_pop_arr[i][-1], ref=self.ref_point)
            if tmp > max_hyp:
                max_hyp = tmp
                pass
            pass
        self.ax_hypv.set_ylim(min_hyp, max_hyp * 1.1)
        self.ax_hypv.grid(True)
        self.ax_hypv.title.set_text('Hypervolume')

    def __call__(self, j):
        pos = j
        first_front_len_arr = []
        first_front_str = '1-st front = '
        for i in range(0, len(self.all_pop_arr)):
            k_0_arr = []
            k_1_arr = []
            pop = self.all_pop_arr[i][pos]
            self.hypervolume_arr_list[i].append(calc_hypervolume_java(pop, num_test=1000000)[1])
            # self.hypervolume_arr_list[i].append(deap.benchmarks.tools.hypervolume(pop, ref=self.ref_point))
            for el in pop:
                k_0_arr.append(el.fitness.values[self.k_0])
                k_1_arr.append(el.fitness.values[self.k_1])
                pass
            self.line_arr[i].set_offsets(np.array([k_0_arr, k_1_arr]).T)

            self.line_hypv_arr[i].set_data(np.arange(len(self.hypervolume_arr_list[i])), self.hypervolume_arr_list[i])
            first_front_len_val = len(deap.tools.emo.sortNondominated(pop, len(pop), first_front_only=True)[0])
            first_front_len_arr.append(first_front_len_val)
            first_front_str += str(first_front_len_val) + "  /  "
            pass

        # self.line.set_data(k_0_arr, k_1_arr)
        # self.line.set_offsets(np.array([k_0_arr, k_1_arr]).T)
        self.gen_text.set_text('Generation {}'.format(j))

        # first_front_str = '1-st front = '
        self.front_size_text.set_text(first_front_str)
        # self.front_size_text.set_text('1-st front = {}, {:.2f}%'.format(first_front_len, first_front_len/len(pop)*100))

        # self.line_hypv.set_data([1,2,3], [7*(10**7), 8*(10**7), 9*(10**7)])
        # self.line_hypv.set_data(np.arange(len(self.hypervolume_arr)), self.hypervolume_arr)
        # self.line_hypv_arr[0].set_data(np.arange(len(self.hypervolume_arr)), self.hypervolume_arr)
        # self.line_hypv_arr[1].set_data(np.arange(len(self.hypervolume_arr)), np.array(self.hypervolume_arr)*1.01)

        # self.line_hypv.set_data(k_0_arr, k_1_arr)
        return self.line_arr[0],

    pass


def calc_avg_dist_diag(front):
    dist_diag = []
    for el in front:
        point = np.array(el.fitness.values)
        dist_point = np.array([sum(point) / len(point) for i in range(0, len(point))])
        dist_diag.append(np.linalg.norm(point - dist_point))
        pass
    dist_diag = np.array(dist_diag)
    return dist_diag.mean()


def main(algo='NSGA-2', knapsack_seed=0, pop_seed=0, n_i=250, n_k=5, pop_size=200, NGEN=10, knapsack=None,
         is_unique_solutions=False, num_iter_repair=0, mating_algo='uniform', sel_for_mating='random',
         freq_cald_hypervolume=10, is_save_all_gen=False,
         ):
    # initialization
    mating_prob = 0.8
    mutate_prob = 0.01
    if knapsack is None:
        np.random.seed(knapsack_seed)
        knapsack = create_knapsack(n_i=n_i, n_k=n_k)
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0,) * n_k)
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Structure initializers
    toolbox.register("individual", create_individual, n_i, knapsack)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_knapsack, knapsack)
    if mating_algo == 'uniform':
        toolbox.register("mate", mate_f_prob, knapsack, tools.cxUniform, 0.5)
    else:
        toolbox.register("mate", mate_f, knapsack, tools.cxOnePoint)
        pass
    toolbox.register("mutate", mutate_f, knapsack, tools.mutFlipBit, mutate_prob)
    if algo == 'NSGA-2':
        toolbox.register("select", tools.selNSGA2, nd='standard',)
        pass
    elif algo == 'PO-count':
        toolbox.register("select", tools.selPO)
        pass
    elif algo == 'AR':
        toolbox.register("select", tools.emo.selAR)
        pass
    elif algo == 'WS':
        toolbox.register("select", tools.emo.selWS)
        pass
    elif algo == 'PO-prob' or algo == 'PO-prob-repair':
        toolbox.register("select", tools.selPO, nd='prob',)
        pass
    elif algo == 'NSGA-3':
        p_arr = [250, 21, 9, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, ]
        P = p_arr[n_k - 2]
        ref_points_nsga3 = deap.tools.uniform_reference_points(n_k, P)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points_nsga3, nd='standard',)
        print('Num ref points = {}'.format(len(ref_points_nsga3)))
        pass

    # stats creation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    logbook = tools.Logbook()
    #старая версия для подсчета только времени
    if freq_cald_hypervolume < NGEN:
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        logbook.header = "gen", "evals", "std", "min", "avg", "max", "hypervol", "dist_diag", "time", 'firts_front'
    else:
        logbook.header = "gen", "hypervol", "time"
        pass

    # generate init population
    np.random.seed(pop_seed)
    pop = toolbox.population(n=pop_size)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    s = time.time()
    pop = toolbox.select(pop, len(pop))
    tot_time = time.time() - s

    front_1 = deap.tools.emo.sortNondominated(pop, len(pop), first_front_only=True)[0]
    hypervol_val = calc_hypervolume_java(front_1)[1]
    # hypervol_val = 1
    if freq_cald_hypervolume < NGEN:
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), hypervol=hypervol_val, dist_diag=calc_avg_dist_diag(front_1),
                   time=tot_time, firts_front=len(front_1), **record)
    else:
        logbook.record(gen=0, hypervol=hypervol_val, time=tot_time)

    # Begin the generational process
    # save population, if required
    if is_save_all_gen:
        pop_arr = [copy.deepcopy(pop)]
    else:
        pop_arr = None
        pass

    for gen in range(1, NGEN):
        if gen >= 50:
            pass
        print (gen)
        if algo == 'PO-prob-repair' and gen == (NGEN-num_iter_repair):
            # repair last num_iter_repair generations
            toolbox.register("select", tools.selNSGA2, nd='standard',)
            pass
        # print(gen)
        # Vary the population
        # offspring = tools.selTournament(pop, len(pop), tournsize=2)
        if sel_for_mating == 'random':
            offspring = tools.selRandom(pop, len(pop))
        if sel_for_mating == 'uniform':
            tmp = tools.selBest(pop, int(len(pop)*0.3))
            offspring = tools.selRandom(tmp, len(pop))
            pass
        elif sel_for_mating == 'best':
            offspring = tools.selBest(pop, len(pop))
            pass
        elif sel_for_mating == 'tournament':
            offspring = tools.selTournament(pop, len(pop), tournsize=2)
            pass
        # record starting time
        offspring = [toolbox.clone(ind) for ind in offspring]

        if is_unique_solutions:
            pop_str_set = set([str(pop[i]) for i in range(0, len(pop))])

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(ind1, ind2)
            # if random.random() <= mating_prob:
            #    toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)

            if is_unique_solutions:
                # mutate, until we get new idividual, not like before
                while str(ind1) in pop_str_set:
                    toolbox.mutate(ind1)
                    pass
                pop_str_set.add(str(ind1))
            pass

            toolbox.mutate(ind2)
            if is_unique_solutions:
                # mutate, until we get new idividual, not like before
                while str(ind2) in pop_str_set:
                    toolbox.mutate(ind2)
                    pass
                pop_str_set.add(str(ind2))
            pass

            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        s = time.time()
        pop = toolbox.select(pop + offspring, pop_size)
        tot_time = time.time() - s

        if freq_cald_hypervolume < NGEN:
            record = stats.compile(pop)
            front_1 = deap.tools.emo.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if gen % freq_cald_hypervolume == 0 or gen == NGEN-1:
                hypervol_val = calc_hypervolume_java(front_1)[1]
                pass
            logbook.record(gen=gen, evals=len(invalid_ind), hypervol=hypervol_val, dist_diag=calc_avg_dist_diag(front_1),
                       time=tot_time, firts_front=len(front_1), **record)
            pass
        else:
            if gen == NGEN-1:
                front_1 = deap.tools.emo.sortNondominated(pop, len(pop), first_front_only=True)[0]
                # hypervol_val = calc_hypervolume_java(front_1)[1]
                hypervol_val = 1
                pass
            logbook.record(gen=gen, hypervol=hypervol_val, time=tot_time)
            pass
        if is_save_all_gen:
            pop_arr.append(copy.deepcopy(pop))
            pass
        pass
    return pop, logbook, pop_arr

