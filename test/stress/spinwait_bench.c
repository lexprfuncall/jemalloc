/*
 * Microbenchmark the CPU hint instruction used by jemalloc's spin loops:
 *
 *   - x86/x86_64: pause
 *   - aarch64/arm: isb
 *
 * Also benchmark a blocking handoff path using pthread condvar ping-pong so
 * that we can compare spinning cost with the cost of sleeping / being woken.
 *
 * Build examples:
 *   cc -O3 -march=native -std=c11 -Wall -Wextra -pthread \
 *       -o spinwait_bench test/stress/spinwait_bench.c
 *
 * Run examples:
 *   ./spinwait_bench
 *   ./spinwait_bench -n 200000000 -c 100000 -r 15 -s 600
 *   ./spinwait_bench -p 0 -A 0 -B 1
 */

#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(__i386__)
#  define SPINWAIT_INSN_NAME "pause"
static inline void
spinwait_insn(void) {
	__asm__ volatile("pause");
}
#elif defined(__aarch64__) || defined(__arm__)
#  define SPINWAIT_INSN_NAME "isb"
static inline void
spinwait_insn(void) {
	__asm__ volatile("isb" ::: "memory");
}
#else
#  error "Unsupported architecture for this benchmark"
#endif

static inline void
baseline_insn(void) {
	/*
	 * Compiler barrier only: preserves the structure of the loop without
	 * issuing the target instruction.
	 */
	__asm__ volatile("" ::: "memory");
}

static double
timespec_to_ns(const struct timespec *ts) {
	return (double)ts->tv_sec * 1e9 + (double)ts->tv_nsec;
}

static double
clock_now_ns(void) {
	struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
	if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) == 0) {
		return timespec_to_ns(&ts);
	}
#endif
	if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
		perror("clock_gettime");
		exit(1);
	}
	return timespec_to_ns(&ts);
}

typedef struct {
	bool available;
	uint64_t begin;
	uint64_t end;
	uint64_t freq_hz;
} cycle_sample_t;

#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t
read_cycles_x86_begin(void) {
	unsigned hi, lo;
	__asm__ volatile("lfence\n\t"
	                 "rdtsc"
	                 : "=a"(lo), "=d"(hi)
	                 :
	                 : "memory");
	return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t
read_cycles_x86_end(void) {
	unsigned hi, lo;
	__asm__ volatile("rdtscp\n\t"
	                 "lfence"
	                 : "=a"(lo), "=d"(hi)
	                 :
	                 : "rcx", "memory");
	return ((uint64_t)hi << 32) | lo;
}
#elif defined(__aarch64__)
static inline uint64_t
read_cntvct(void) {
	uint64_t vct;
	__asm__ volatile("isb\n\t"
	                 "mrs %0, cntvct_el0"
	                 : "=r"(vct));
	return vct;
}

static inline uint64_t
read_cntfrq(void) {
	uint64_t frq;
	__asm__ volatile("mrs %0, cntfrq_el0" : "=r"(frq));
	return frq;
}
#endif

static inline cycle_sample_t
read_cycles_begin(void) {
	cycle_sample_t s;
	memset(&s, 0, sizeof(s));
#if defined(__x86_64__) || defined(__i386__)
	s.available = true;
	s.begin = read_cycles_x86_begin();
#elif defined(__aarch64__)
	s.available = true;
	s.freq_hz = read_cntfrq();
	s.begin = read_cntvct();
#else
	s.available = false;
#endif
	return s;
}

static inline void
read_cycles_end(cycle_sample_t *s) {
	if (!s->available) {
		return;
	}
#if defined(__x86_64__) || defined(__i386__)
	s->end = read_cycles_x86_end();
#elif defined(__aarch64__)
	s->end = read_cntvct();
#endif
}

static inline uint64_t
cycle_delta(const cycle_sample_t *s) {
	return s->end - s->begin;
}

static int
double_cmp(const void *a, const void *b) {
	double da = *(const double *)a;
	double db = *(const double *)b;
	return (da > db) - (da < db);
}

static int
u64_cmp(const void *a, const void *b) {
	uint64_t ua = *(const uint64_t *)a;
	uint64_t ub = *(const uint64_t *)b;
	return (ua > ub) - (ua < ub);
}

static void
pin_process_to_cpu_or_die(int cpu) {
#if defined(__linux__)
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET((unsigned)cpu, &set);
	if (sched_setaffinity(0, sizeof(set), &set) != 0) {
		fprintf(stderr, "sched_setaffinity(%d) failed: %s\n",
		    cpu, strerror(errno));
		exit(1);
	}
#else
	(void)cpu;
	fprintf(stderr, "CPU pinning is only implemented on Linux in this tool.\n");
	exit(1);
#endif
}

static void
pin_current_thread_to_cpu_or_die(int cpu) {
	if (cpu < 0) {
		return;
	}
#if defined(__linux__)
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET((unsigned)cpu, &set);
	if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
		fprintf(stderr, "pthread_setaffinity_np(%d) failed: %s\n",
		    cpu, strerror(errno));
		exit(1);
	}
#else
	(void)cpu;
	fprintf(stderr,
	    "Thread CPU pinning is only implemented on Linux in this tool.\n");
	exit(1);
#endif
}

static void
usage(const char *prog) {
	fprintf(stderr,
	    "Usage: %s [-n spin_iterations] [-c ctx_roundtrips] [-r repeats]\n"
	    "          [-s jemalloc_spin_count] [-p cpu] [-A main_cpu] [-B worker_cpu]\n",
	    prog);
}

#define UNROLL8(stmt)                                                         \
	do {                                                                  \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
		stmt;                                                         \
	} while (0)

__attribute__((noinline)) static void
run_baseline(uint64_t iters) {
	uint64_t i = 0;
	for (; i + 8 <= iters; i += 8) {
		UNROLL8(baseline_insn());
	}
	for (; i < iters; i++) {
		baseline_insn();
	}
}

__attribute__((noinline)) static void
run_spinwait(uint64_t iters) {
	uint64_t i = 0;
	for (; i + 8 <= iters; i += 8) {
		UNROLL8(spinwait_insn());
	}
	for (; i < iters; i++) {
		spinwait_insn();
	}
}

typedef struct {
	double ns;
	uint64_t cycles;
	bool cycles_available;
	uint64_t cycles_freq_hz;
} timing_t;

static timing_t
measure_once(void (*fn)(uint64_t), uint64_t iters) {
	timing_t t;
	memset(&t, 0, sizeof(t));

	cycle_sample_t c = read_cycles_begin();
	double start = clock_now_ns();
	fn(iters);
	double stop = clock_now_ns();
	read_cycles_end(&c);

	t.ns = stop - start;
	t.cycles_available = c.available;
	t.cycles = c.available ? cycle_delta(&c) : 0;
	t.cycles_freq_hz = c.freq_hz;
	return t;
}

static void
warm_up(uint64_t iters) {
	run_baseline(iters);
	run_spinwait(iters);
}

typedef struct {
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	bool started;
	bool stop;
	int turn;
	uint64_t roundtrips_done;
	int cpu_worker;
} pingpong_state_t;

static void *
pingpong_worker(void *opaque) {
	pingpong_state_t *state = (pingpong_state_t *)opaque;
	pin_current_thread_to_cpu_or_die(state->cpu_worker);

	if (pthread_mutex_lock(&state->mutex) != 0) {
		perror("pthread_mutex_lock");
		exit(1);
	}
	while (!state->started) {
		if (pthread_cond_wait(&state->cond, &state->mutex) != 0) {
			perror("pthread_cond_wait");
			exit(1);
		}
	}

	while (!state->stop) {
		while (state->turn != 1 && !state->stop) {
			if (pthread_cond_wait(&state->cond, &state->mutex) != 0) {
				perror("pthread_cond_wait");
				exit(1);
			}
		}
		if (state->stop) {
			break;
		}
		state->turn = 0;
		state->roundtrips_done++;
		if (pthread_cond_signal(&state->cond) != 0) {
			perror("pthread_cond_signal");
			exit(1);
		}
	}

	if (pthread_mutex_unlock(&state->mutex) != 0) {
		perror("pthread_mutex_unlock");
		exit(1);
	}
	return NULL;
}

static timing_t
measure_context_switch_once(uint64_t roundtrips, int cpu_main, int cpu_worker) {
	timing_t t;
	memset(&t, 0, sizeof(t));

	pingpong_state_t state;
	memset(&state, 0, sizeof(state));
	state.cpu_worker = cpu_worker;
	if (pthread_mutex_init(&state.mutex, NULL) != 0) {
		perror("pthread_mutex_init");
		exit(1);
	}
	if (pthread_cond_init(&state.cond, NULL) != 0) {
		perror("pthread_cond_init");
		exit(1);
	}

	pthread_t worker;
	if (pthread_create(&worker, NULL, pingpong_worker, &state) != 0) {
		perror("pthread_create");
		exit(1);
	}

	pin_current_thread_to_cpu_or_die(cpu_main);

	if (pthread_mutex_lock(&state.mutex) != 0) {
		perror("pthread_mutex_lock");
		exit(1);
	}

	state.started = true;
	state.turn = 0;
	if (pthread_cond_signal(&state.cond) != 0) {
		perror("pthread_cond_signal");
		exit(1);
	}

	cycle_sample_t c = read_cycles_begin();
	double start = clock_now_ns();
	for (uint64_t i = 0; i < roundtrips; i++) {
		state.turn = 1;
		if (pthread_cond_signal(&state.cond) != 0) {
			perror("pthread_cond_signal");
			exit(1);
		}
		while (state.turn != 0) {
			if (pthread_cond_wait(&state.cond, &state.mutex) != 0) {
				perror("pthread_cond_wait");
				exit(1);
			}
		}
	}
	double stop = clock_now_ns();
	read_cycles_end(&c);

	state.stop = true;
	if (pthread_cond_signal(&state.cond) != 0) {
		perror("pthread_cond_signal");
		exit(1);
	}
	if (pthread_mutex_unlock(&state.mutex) != 0) {
		perror("pthread_mutex_unlock");
		exit(1);
	}

	if (pthread_join(worker, NULL) != 0) {
		perror("pthread_join");
		exit(1);
	}

	if (state.roundtrips_done != roundtrips) {
		fprintf(stderr,
		    "unexpected roundtrip count: expected=%" PRIu64
		    " observed=%" PRIu64 "\n",
		    roundtrips, state.roundtrips_done);
		exit(1);
	}

	if (pthread_cond_destroy(&state.cond) != 0) {
		perror("pthread_cond_destroy");
		exit(1);
	}
	if (pthread_mutex_destroy(&state.mutex) != 0) {
		perror("pthread_mutex_destroy");
		exit(1);
	}

	t.ns = stop - start;
	t.cycles_available = c.available;
	t.cycles = c.available ? cycle_delta(&c) : 0;
	t.cycles_freq_hz = c.freq_hz;
	return t;
}

static void
warm_up_context_switch(uint64_t roundtrips, int cpu_main, int cpu_worker) {
	(void)measure_context_switch_once(roundtrips, cpu_main, cpu_worker);
}

int
main(int argc, char **argv) {
	uint64_t iterations = 100000000;
	uint64_t context_roundtrips = 100000;
	int repeats = 11;
	int jemalloc_spin_count = 600;
	int cpu = -1;
	int ctx_cpu_main = -1;
	int ctx_cpu_worker = -1;

	for (int opt; (opt = getopt(argc, argv, "n:c:r:s:p:A:B:h")) != -1;) {
		switch (opt) {
		case 'n':
			iterations = strtoull(optarg, NULL, 10);
			break;
		case 'c':
			context_roundtrips = strtoull(optarg, NULL, 10);
			break;
		case 'r':
			repeats = atoi(optarg);
			break;
		case 's':
			jemalloc_spin_count = atoi(optarg);
			break;
		case 'p':
			cpu = atoi(optarg);
			break;
		case 'A':
			ctx_cpu_main = atoi(optarg);
			break;
		case 'B':
			ctx_cpu_worker = atoi(optarg);
			break;
		case 'h':
		default:
			usage(argv[0]);
			return opt == 'h' ? 0 : 1;
		}
	}

	if (iterations == 0 || context_roundtrips == 0 || repeats <= 0 ||
	    jemalloc_spin_count < 0) {
		usage(argv[0]);
		return 1;
	}

	if (cpu >= 0) {
		pin_process_to_cpu_or_die(cpu);
	}

	double *baseline_ns = calloc((size_t)repeats, sizeof(double));
	double *spin_ns = calloc((size_t)repeats, sizeof(double));
	double *delta_ns = calloc((size_t)repeats, sizeof(double));
	double *ctx_ns = calloc((size_t)repeats, sizeof(double));
	uint64_t *baseline_cycles = calloc((size_t)repeats, sizeof(uint64_t));
	uint64_t *spin_cycles = calloc((size_t)repeats, sizeof(uint64_t));
	uint64_t *delta_cycles = calloc((size_t)repeats, sizeof(uint64_t));
	uint64_t *ctx_cycles = calloc((size_t)repeats, sizeof(uint64_t));
	if (baseline_ns == NULL || spin_ns == NULL || delta_ns == NULL ||
	    ctx_ns == NULL || baseline_cycles == NULL || spin_cycles == NULL ||
	    delta_cycles == NULL || ctx_cycles == NULL) {
		fprintf(stderr, "allocation failure\n");
		return 1;
	}

	warm_up(iterations / 10 > 0 ? iterations / 10 : iterations);
	warm_up_context_switch(context_roundtrips / 10 > 0
	        ? context_roundtrips / 10
	        : context_roundtrips,
	    ctx_cpu_main, ctx_cpu_worker);

	bool have_cycles = false;
	uint64_t cycle_freq_hz = 0;
	for (int i = 0; i < repeats; i++) {
		timing_t b = measure_once(run_baseline, iterations);
		timing_t s = measure_once(run_spinwait, iterations);
		timing_t c = measure_context_switch_once(context_roundtrips,
		    ctx_cpu_main, ctx_cpu_worker);

		baseline_ns[i] = b.ns;
		spin_ns[i] = s.ns;
		delta_ns[i] = s.ns - b.ns;
		ctx_ns[i] = c.ns;

		if (b.cycles_available && s.cycles_available && c.cycles_available) {
			have_cycles = true;
			baseline_cycles[i] = b.cycles;
			spin_cycles[i] = s.cycles;
			delta_cycles[i] = s.cycles - b.cycles;
			ctx_cycles[i] = c.cycles;
			if (c.cycles_freq_hz != 0) {
				cycle_freq_hz = c.cycles_freq_hz;
			} else if (s.cycles_freq_hz != 0) {
				cycle_freq_hz = s.cycles_freq_hz;
			}
		}
	}

#if !defined(__aarch64__)
	(void)cycle_freq_hz;
#endif

	qsort(baseline_ns, (size_t)repeats, sizeof(double), double_cmp);
	qsort(spin_ns, (size_t)repeats, sizeof(double), double_cmp);
	qsort(delta_ns, (size_t)repeats, sizeof(double), double_cmp);
	qsort(ctx_ns, (size_t)repeats, sizeof(double), double_cmp);
	if (have_cycles) {
		qsort(baseline_cycles, (size_t)repeats, sizeof(uint64_t), u64_cmp);
		qsort(spin_cycles, (size_t)repeats, sizeof(uint64_t), u64_cmp);
		qsort(delta_cycles, (size_t)repeats, sizeof(uint64_t), u64_cmp);
		qsort(ctx_cycles, (size_t)repeats, sizeof(uint64_t), u64_cmp);
	}

	int median = repeats / 2;
	double median_baseline_ns_per_iter = baseline_ns[median] / (double)iterations;
	double median_spin_ns_per_iter = spin_ns[median] / (double)iterations;
	double median_delta_ns_per_iter = delta_ns[median] / (double)iterations;
	double median_ctx_ns_per_roundtrip =
	    ctx_ns[median] / (double)context_roundtrips;
	double median_ctx_ns_per_handoff = median_ctx_ns_per_roundtrip / 2.0;
	double estimated_spin_budget_ns =
	    median_delta_ns_per_iter * (double)jemalloc_spin_count;

	printf("spinwait benchmark\n");
	printf("  arch instruction : %s\n", SPINWAIT_INSN_NAME);
	printf("  spin iterations  : %" PRIu64 "\n", iterations);
	printf("  ctx roundtrips   : %" PRIu64 "\n", context_roundtrips);
	printf("  repeats          : %d\n", repeats);
	if (cpu >= 0) {
		printf("  spin pinned cpu  : %d\n", cpu);
	}
	if (ctx_cpu_main >= 0 || ctx_cpu_worker >= 0) {
		printf("  ctx main cpu     : %d\n", ctx_cpu_main);
		printf("  ctx worker cpu   : %d\n", ctx_cpu_worker);
	}
	printf("\n");

	printf("median wall-clock time\n");
	printf("  baseline loop        : %.6f ns/iter\n",
	    median_baseline_ns_per_iter);
	printf("  %s loop          : %.6f ns/iter\n",
	    SPINWAIT_INSN_NAME, median_spin_ns_per_iter);
	printf("  delta (%s only)      : %.6f ns/iter\n",
	    SPINWAIT_INSN_NAME, median_delta_ns_per_iter);
	printf("\n");

	printf("jemalloc calibration\n");
	printf("  mutex_max_spin       : %d\n", jemalloc_spin_count);
	printf("  est. time in %d x %s : %.3f ns (%.3f us)\n",
	    jemalloc_spin_count, SPINWAIT_INSN_NAME,
	    estimated_spin_budget_ns,
	    estimated_spin_budget_ns / 1000.0);
	printf("\n");

	printf("context-switch / wakeup benchmark\n");
	printf("  pthread condvar roundtrip : %.3f ns (%.3f us)\n",
	    median_ctx_ns_per_roundtrip, median_ctx_ns_per_roundtrip / 1000.0);
	printf("  est. per handoff         : %.3f ns (%.3f us)\n",
	    median_ctx_ns_per_handoff, median_ctx_ns_per_handoff / 1000.0);
	if (estimated_spin_budget_ns > 0.0) {
		printf("  handoff / spin budget    : %.2f x\n",
		    median_ctx_ns_per_handoff / estimated_spin_budget_ns);
	}
	printf("\n");

	if (have_cycles) {
		double median_baseline_cycles_per_iter =
		    (double)baseline_cycles[median] / (double)iterations;
		double median_spin_cycles_per_iter =
		    (double)spin_cycles[median] / (double)iterations;
		double median_delta_cycles_per_iter =
		    (double)delta_cycles[median] / (double)iterations;
		double median_ctx_cycles_per_roundtrip =
		    (double)ctx_cycles[median] / (double)context_roundtrips;
		double median_ctx_cycles_per_handoff =
		    median_ctx_cycles_per_roundtrip / 2.0;

		printf("median cycle-counter time\n");
#if defined(__aarch64__)
		if (cycle_freq_hz != 0) {
			printf("  counter freq         : %" PRIu64 " Hz\n", cycle_freq_hz);
		}
#endif
		printf("  baseline loop        : %.6f cycles/iter\n",
		    median_baseline_cycles_per_iter);
		printf("  %s loop          : %.6f cycles/iter\n",
		    SPINWAIT_INSN_NAME, median_spin_cycles_per_iter);
		printf("  delta (%s only)      : %.6f cycles/iter\n",
		    SPINWAIT_INSN_NAME, median_delta_cycles_per_iter);
		printf("  est. cycles in %d x %s : %.3f cycles\n",
		    jemalloc_spin_count, SPINWAIT_INSN_NAME,
		    median_delta_cycles_per_iter * (double)jemalloc_spin_count);
		printf("  ctx roundtrip        : %.3f cycles\n",
		    median_ctx_cycles_per_roundtrip);
		printf("  ctx handoff          : %.3f cycles\n",
		    median_ctx_cycles_per_handoff);
		printf("\n");
	}

	printf("notes\n");
	printf("  - The spin delta subtracts the loop/control overhead from the\n");
	printf("    measured instruction loop.\n");
	printf("  - This estimates only spin_cpu_spinwait(); jemalloc's contended\n");
	printf("    slow path also does loads, branches, and sometimes trylock work.\n");
	printf("  - The context benchmark is not a pure kernel context-switch cost;\n");
	printf("    it measures a pthread condvar sleep+wakeup+schedule handoff.\n");
	printf("    That is closer to the 'stop spinning and block' tradeoff.\n");
	printf("  - For more stable numbers: pin to CPUs, run on an otherwise idle\n");
	printf("    machine, and keep iteration counts large.\n");

	free(baseline_ns);
	free(spin_ns);
	free(delta_ns);
	free(ctx_ns);
	free(baseline_cycles);
	free(spin_cycles);
	free(delta_cycles);
	free(ctx_cycles);

	return 0;
}
