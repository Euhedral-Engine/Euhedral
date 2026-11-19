/* Harness of the programmatic-launch registry: the registry source is included, so the harness reads its
 * `used` count and `LIMIT`. Exit status 0 when every check holds. */
#include REGISTRY_SOURCE
#include <pthread.h>
#include <stdio.h>

#define THREADS 8
#define SHARED_KEYS 500
#define ROUNDS 20

static pthread_barrier_t start;

static uintptr_t key_of(unsigned int index) { return 0x10000u + 0x40u * (uintptr_t)index; }

static void* register_shared(void* argument) {
    const unsigned int offset = (unsigned int)(uintptr_t)argument;
    pthread_barrier_wait(&start);
    /* Every thread registers every key, each from its own starting point, so the same slots race. */
    for (unsigned int round = 0; round < ROUNDS; round++)
        for (unsigned int index = 0; index < SHARED_KEYS; index++)
            euhedral_pdl_registry_add((const void*)key_of((index + offset * 61u) % SHARED_KEYS));
    return NULL;
}

static int check(int condition, const char* what) {
    if (!condition) fprintf(stderr, "FAIL: %s\n", what);
    return condition ? 0 : 1;
}

int main(void) {
    int failures = 0;
    pthread_t threads[THREADS];
    pthread_barrier_init(&start, NULL, THREADS);
    for (unsigned int t = 0; t < THREADS; t++) pthread_create(&threads[t], NULL, register_shared, (void*)(uintptr_t)t);
    for (unsigned int t = 0; t < THREADS; t++) pthread_join(threads[t], NULL);
    unsigned int missing = 0;
    for (unsigned int index = 0; index < SHARED_KEYS; index++)
        if (!euhedral_pdl_registry_contains((const void*)key_of(index))) missing++;
    failures += check(missing == 0, "every shared key is found");
    failures += check(atomic_load(&used) == SHARED_KEYS, "a duplicate registration takes no second slot");
    failures += check(!euhedral_pdl_registry_contains((const void*)key_of(SHARED_KEYS)), "an unregistered key is absent");

    /* Fill past the limit from one thread: exactly LIMIT registrations in all are accepted. */
    unsigned int accepted = SHARED_KEYS;
    for (unsigned int index = SHARED_KEYS; index < 4000; index++)
        if (euhedral_pdl_registry_add((const void*)key_of(index))) accepted++;
    failures += check(accepted == LIMIT, "a full registry accepts exactly its limit");
    failures += check(atomic_load(&used) == LIMIT, "the used count stops at the limit");
    return failures == 0 ? 0 : 1;
}
