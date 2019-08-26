import numpy as np
import sys
import os
import time
from functools import partial, reduce
from itertools import permutations, chain

# Plots
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm, trange

# Parallelising fitness
from multiprocessing import Pool, cpu_count

# Used to suppress console output
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class Genus():
    ''' Storing information about all the possible gene combinations.

    INPUT
        (kwargs) genomes
    '''

    def __init__(self, **genomes):
        self.__dict__.update(
            {key : np.asarray(val) for (key, val) in genomes.items()}
            )

    def create_organisms(self, amount = 1):
        ''' Create organisms of this genus.
        
        INPUT
            (int) amount = 1
        '''
        organisms = np.array([Organism(genus = self, 
            **{key : val[np.random.choice(range(val.shape[0]))]
            for (key, val) in self.__dict__.items()}) for _ in range(amount)])
        return organisms

    def alter_genomes(self, **genomes):
        ''' Add or change genomes to the genus.
        
        INPUT
            (kwargs) genomes
        '''
        self.__dict__.update(genomes)
        return self

    def remove_genomes(self, *keys):
        ''' Remove genomes from the genus. '''
        for key in keys:
            self.__dict__.pop(key, None)
        return self

class Organism():
    ''' Organism of a particular genus. 

    INPUT
        (Genus) genus
        (kwargs) genome: genome information
    '''

    def __init__(self, genus, **genome):

        # Check that the input parameters match with the genus type,
        # and if any parameters are missing then add random values
        genome = {key : val for (key, val) in genome.items() if key in
            genus.__dict__.keys() and val in genus.__dict__[key]}
        for key in genus.__dict__.keys() - genome.keys():
            val_idx = np.random.choice(range(genus.__dict__[key].shape[0]))
            genome[key] = genus.__dict__[key][val_idx]

        self.__dict__.update(genome)
        self.genus = genus

    def get_genome(self):
        attrs = self.__dict__.items()
        return {key : val for (key, val) in attrs if key != 'genus'}

    def breed(self, other):
        ''' Breed organism with another organism, returning a new
            organism of the same genus.

        INPUT
            (Organism) other
        '''

        if self.genus != other.genus:
            raise Exception("Only organisms of the same genus can breed.")

        # Child will inherit genes from its parents randomly
        parents_genomes = {
            key : (self.get_genome()[key], other.get_genome()[key])
                for key in self.get_genome().keys()
            }
        child_genome = {
            key : pair[np.random.choice([0, 1])]
                for (key, pair) in parents_genomes.items()
        }

        return Organism(self.genus, **child_genome)

    def mutate(self):
        ''' Return mutated version of the organism, where the mutated version
            will on average have one gene different from the original. '''
        keys = np.asarray(list(self.get_genome().keys()))
        mut_idx = np.less(np.random.random(keys.size), np.divide(1, keys.size))
        mut_vals = {key : self.genus.__dict__[key]\
            [np.random.choice(range(self.genus.__dict__[key].shape[0]))]
            for key in keys[mut_idx]}
        self.__dict__.update(mut_vals)
        return self

class Population():
    ''' Population of organisms, all of the same genus.

    INPUT
        (Genus) genus
        (int) size
        (function) fitness_fn
        (function) post_fn: Function to be applied to fitness values in plots
        (dict) initial_genome = None: this will construct a homogenous
               population only consisting of the initial genome, for a
               warm start
        '''

    def __init__(self, genus, size, fitness_fn, post_fn = lambda x: x,
        initial_genome = None):

        self.genus = genus
        self.size = size
        self.post_fn = post_fn
        
        # Fitness function must be pickleable, so in particular it
        # cannot be a lambda expression
        self.fitness_fn = fitness_fn

        if initial_genome:
            self.population = np.array(
                [Organism(genus, **initial_genome) for _ in range(size)])
        else:
            self.population = genus.create_organisms(size)

    def get_genomes(self):
        return np.asarray([o.get_genome() for o in self.population])

    def get_fitness(self, multiprocessing = True, workers = cpu_count(),
        progress_bar = True, history = None, generation = None):
        ''' Compute fitness values of population.

        INPUT
            (bool) multiprocessing = True: whether fitnesses should be
                   computed in parallel
            (int) workers = cpu_count(): how many workers to use if
                  multiprocessing is True
            (bool) progress_bar = True: show progress bar
            (History) history = None: previous genome and fitness history
            (int) generation = None

        OUTPUT
            (ndarray) fitness values
        '''

        pop = self.population
        fitnesses = np.zeros(pop.size)

        # Duck typing function to make things immutable
        def make_immutable(x):
            try:
                if not isinstance(x, str):
                    x = tuple(x)
            except TypeError:
                pass
            return x
        def immute_dict(d):
            return {key : make_immutable(val) for (key, val) in d.items()}

        unique_genomes = np.array([dict(dna) for dna in
            set(frozenset(immute_dict(genome).items())
            for genome in self.get_genomes())
            ])

        # If history is loaded then get the genomes from the current
        # population that are unique across all generations
        past_indices = np.array([])
        if history and generation:
            g_prev = history.genome_history
            f_prev = history.fitness_history

            indices = np.array([((np.where(g_prev == org.get_genome())[0][0],
                np.where(g_prev == org.get_genome())[1][0]), idx)
                for (idx, org) in enumerate(pop)
                if org.get_genome() in g_prev
                ])
            past_indices = np.array([idx for (_, idx) in indices])

            # Load previous fitnesses of genomes that are occuring now
            for (past_idx, idx) in indices:
                fitnesses[idx] = f_prev[generation-past_idx[0]-1, past_idx[1]]

            # Remove genomes that have occured previously
            unique_genomes = np.array([genome for genome in unique_genomes
                if genome not in g_prev])

        # Pull out the organisms with the unique genomes
        unique_indices = np.array([
            np.min(np.array([idx for (idx, org) in enumerate(pop)
                if immute_dict(org.get_genome()) == immute_dict(genome)
                ]))
            for genome in unique_genomes
            ])

        # If there are any organisms whose fitness we didn't already
        # know then compute them
        if unique_indices.size:
            unique_orgs = pop[unique_indices]

            if isinstance(generation, int):
                progress_text = f"Computing fitness for gen {generation}"
            else:
                progress_text = f"Computing fitness"

            # Compute fitness values without computing the same one twice
            fn = self.fitness_fn
            with suppress_stdout():
                if multiprocessing:
                    with Pool(workers) as pool:
                        if progress_bar:
                            fit_iter = tqdm(zip(unique_indices, 
                                pool.imap(fn, unique_orgs)),
                                total = unique_orgs.size)
                            fit_iter.set_description(progress_text)
                        else:
                            fit_iter = zip(unique_indices,
                                pool.map(fn, unique_orgs))
                        for (i, new_fitness) in fit_iter:
                            fitnesses[i] = new_fitness
                else:
                    if progress_bar:
                        fit_iter = tqdm(zip(unique_indices,
                            map(fn, unique_orgs)),
                            total = unique_orgs.size)
                        fit_iter.set_description(progress_text)
                    else:
                        fit_iter = zip(unique_indices, map(fn, unique_orgs))
                    for (i, new_fitness) in fit_iter:
                        fitnesses[i] = new_fitness

        # Copy out the fitness values to the other organisms with same genome
        for (i, org) in enumerate(pop):
            if i not in unique_indices and i not in past_indices:
                prev_unique_idx = np.min(np.array([idx
                    for idx in unique_indices
                    if immute_dict(org.get_genome()) == \
                        immute_dict(pop[idx].get_genome())
                    ]))
                fitnesses[i] = fitnesses[prev_unique_idx]
    
        return fitnesses


    def sample(self, fitnesses, amount = 1):
        ''' Sample a fixed amount of organisms from the population,
            where the fitter an organism is, the more it's likely
            to be chosen. 
    
        INPUT
            (int) amount = 1: number of organisms to sample

        OUTPUT
            (ndarray) sample of population
        '''

        pop = self.population

        # Convert fitness values into probabilities
        probs = np.divide(fitnesses, sum(fitnesses))

        # Sort the probabilities in descending order and sort the
        # population in the same way
        sorted_idx = np.argsort(probs)[::-1]
        probs = probs[sorted_idx]
        pop = pop[sorted_idx]

        # Get random numbers between 0 and 1 
        indices = np.random.random(amount)

        for i in range(amount):
            # Find the index of the fitness value whose accumulated
            # sum exceeds the value of the i'th random number.
            fn = lambda x, y: (x[0], x[1] + y[1]) \
                              if x[1] + y[1] > indices[i] \
                              else (x[0] + y[0], x[1] + y[1])
            (idx, _) = reduce(fn, map(lambda x: (1, x), probs))
            indices[i] = idx - 1
        
        # Return the organisms indexed at the indices found above
        return pop[indices.astype(int)]

    def evolve(self, generations = 1, breeding_rate = 0.8,
        mutation_rate = 0.2, elitism_rate = 0.05, multiprocessing = True,
        workers = cpu_count(), progress_bars = 2, memory = 20, goal = None,
        verbose = 0):
        ''' Evolve the population.

        INPUT
            (int) generations = 1: number of generations to evolve
            (float) breeding_rate = 0.8: percentage of population to breed 
            (float) mutation_rate = 0.2: percentage of population to mutate
                    each generation
            (float) elitism rate = 0.05: percentage of population to keep
                    across generations
            (bool) multiprocessing = True: whether fitnesses should be
                   computed in parallel
            (int) workers = cpu_count(): how many workers to use if
                  multiprocessing is True
            (int) progress_bars = 2: number of progress bars to show, where 1
                  only shows the main evolution progress, and 2 shows both
                  the evolution and the fitness computation per generation
            (int or string) memory = 20: how many generations the population 
                            can look back to avoid redundant fitness 
                            computations, where 'inf' means unlimited memory.
            (float) goal = None: stop when fitness is above or equal to this
                    value
            (int) verbose = 0: verbosity mode
        '''
    
        history = History(
            population_size = self.size,
            generations = generations,
            memory = memory,
            post_fn = self.post_fn
            )

        if progress_bars:
            gen_iter = trange(generations)
            gen_iter.set_description("Evolving population")
        else:
            gen_iter = range(generations)

        for generation in gen_iter:

            if goal and history.fittest['fitness'] >= goal:
                history.fitness_history = \
                    history.fitness_history[:generation, :]
                # Close tqdm iterator
                if progress_bars:
                    gen_iter.close()
                break

            if verbose >= 2:
                print(f"\n\n~~~ GENERATION {generation} ~~~")

            # Select the portion of the population that will breed
            fitnesses = self.get_fitness(
                multiprocessing = multiprocessing,
                workers = workers,
                progress_bar = (progress_bars >= 2),
                history = history,
                generation = generation
                )

            if verbose >= 2:
                print("\nFitness values:")
                print(fitnesses)

            history.add_entry(
                genomes = self.get_genomes(),
                fitnesses = fitnesses,
                generation = generation
                )

            # Select elites 
            elites_amt = np.ceil(self.size * elitism_rate).astype(int)
            if elitism_rate:
                elites = self.sample(fitnesses, amount = elites_amt)

                if verbose >= 2:
                    print(f"\nElite pool, of size {elites_amt}:")
                    print(np.array([org.get_genome() for org in elites]))

            breeders_amt = max(2, np.ceil(self.size*breeding_rate).astype(int))
            breeders = self.sample(fitnesses, amount = breeders_amt)

            if verbose >= 2:
                print(f"\nBreeding pool, of size {breeders_amt}:")
                print(np.array([org.get_genome() for org in breeders]))
                print("\nBreeding...")

            # Breed until we reach the same size
            children_amt = self.size - elites_amt
            parents = np.random.choice(breeders, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(children_amt)])

            # Find the mutation pool
            mutators = np.less(np.random.random(children_amt), mutation_rate)

            if verbose >= 2:
                print(f"\nMutation pool, of size {children[mutators].size}:")
                print(np.array([c.get_genome() for c in children[mutators]]))
                print("\nMutating...")

            # Mutate the children
            for mutator in children[mutators]:
                mutator.mutate()

            # The children constitutes our new generation
            if elitism_rate:
                self.population = np.append(children, elites)
            else:
                self.population = children
            
            if verbose >= 2:
                print(f"\nNew population, of size {self.population.size}:")
                print(self.get_genomes())
                print(f"\nMean fitness: {np.mean(fitnesses)}")
                print(f"Std fitness: {np.std(fitnesses)}")
                print(f"Fittest so far: {history.fittest}\n")

            if verbose:
                print("Best genome so far:")
                print(history.fittest)

            if verbose >= 3:
                input("Press Enter to continue...")

        # Print a blank line if we're using two progress bars 
        if progress_bars >= 2:
            print("")

        return history

class History():
    ''' History of a population's evolution.
        
        INPUT
            (int) population_size
            (int) generations
            (int or string) memory = 20: how many generations the
                            population can look back to avoid redundant
                            fitness computations, where 'inf' means unlimited
                            memory.
    '''

    def __init__(self, population_size, generations, memory = 20,
        post_fn = lambda x: x):

        self.post_fn = post_fn
        if memory == 'inf' or memory > generations:
            self.memory = generations
        else:
            self.memory = memory
        self.genome_history = np.empty((self.memory, population_size), dict)
        self.fitness_history = np.empty((generations, population_size), float)
        self.fittest = {'genome' : None, 'fitness' : 0}
    
    def add_entry(self, genomes, fitnesses, generation):
        ''' Add genomes and fitnesses to the history. 

        INPUT
            (ndarray) genomes: array of genomes
            (ndarray) fitnesses: array of fitnesses
        '''

        post_fitnesses = np.vectorize(self.post_fn)(fitnesses)
        if max(post_fitnesses) > self.fittest['fitness']:
            self.fittest['genome'] = genomes[np.argmax(post_fitnesses)]
            self.fittest['fitness'] = max(post_fitnesses)

        np.roll(self.genome_history, 1, axis = 0)
        self.genome_history[0, :] = genomes
        self.fitness_history[generation, :] = fitnesses

        return self

    def plot(self, title = 'Fitness by generation', xlabel = 'Generations',
        ylabel = 'Fitness', file_name = None, show_plot = True,
        show_max = True, discrete = False, legend = True,
        legend_location = 'lower right', max_points = 100):
        ''' Plot the fitness values.

        INPUT
            (string) title = 'Fitness by generation'
            (string) xlabel = 'Generations': label on the x-axis
            (string) ylabel = 'Fitness': label on the y-axis
            (string) file_name = None: file name to save the plot to
            (bool) show_plot = True: show plot as a pop-up
            (bool) show_max = True: show max value line on plot
            (bool) discrete = False: make the error plot discrete
            (bool) legend = True: show legend
            (string or int) legend_location = 'lower right': legend location, 
                            either as e.g. 'lower right' or as an integer
                            between 0 and 10
            (int) max_points = 100: maximum number of points on the plot
        '''
        
        fits = np.vectorize(self.post_fn)(self.fitness_history)
        gens = fits.shape[0]
        means = np.mean(fits, axis = 1)
        stds = np.std(fits, axis = 1)
        xs = range(gens)
        if show_max:
            maxs = np.array([np.max(fits[gen, :]) for gen in xs])

        if gens > max_points:
            xs = np.linspace(0, gens - 1, num = max_points).astype(int)

        plt.style.use("ggplot")
        plt.figure()
        plt.xlim(0, gens)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show_max:
            plt.plot(xs, maxs[xs], '--', color = 'blue', label = 'max')

        if discrete:
            plt.errorbar(xs, means[xs], stds[xs], fmt = 'ok', label = 'mean and std')
        else:
            plt.plot(xs, means[xs], '-', color = 'black', label = 'mean')
            plt.fill_between(xs, means[xs] - stds[xs], means[xs] + stds[xs], alpha = 0.2,
                color = 'gray', label = 'std')

        if legend:
            plt.legend(loc = legend_location)

        if file_name:
            plt.savefig(file_name)

        if show_plot:
            plt.show()


def __main__():
    pass
