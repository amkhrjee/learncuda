#include <stdlib.h>
#include "cuda_error.h"
#include <iostream>
#include <iomanip>

#define SIZE (100 * 1024 * 1024)
#define ELEMENTS (SIZE / sizeof(unsigned int))
#define HASH_ENTRIES 1024

void *big_random_block(int size)
{
    unsigned char *data = (unsigned char *)malloc(size);
    HANDLE_NULL(data);
    for (int i = 0; i < size; i++)
        data[i] = rand();

    return data;
}

struct Entry
{
    unsigned int key;
    void *value;
    Entry *next;
};

struct Table
{
    size_t count;
    Entry **entries;
    Entry *pool;
    Entry *firstFree;
};

void initialize_table(Table &table, int entries, int elements)
{
    table.count = entries;
    table.entries = (Entry **)calloc(entries, sizeof(Entry *));
    table.pool = (Entry *)malloc(elements * sizeof(Entry));
    table.firstFree = table.pool;
}

void free_table(Table &table)
{
    free(table.entries);
    free(table.pool);
}

// the simplest hash function possible :-)
size_t hash(unsigned int key, size_t count)
{
    return key % count;
}

void add_to_table(Table &table, unsigned int key, void *value)
{
    size_t hashValue = hash(key, table.count);

    Entry *location = table.firstFree++;
    location->key = key;
    location->value = value;

    location->next = table.entries[hashValue];
    table.entries[hashValue] = location;
}

void verify_table(const Table &table)
{
    int count = 0;
    for (size_t i = 0; i < table.count; i++)
    {
        Entry *current = table.entries[i];
        while (current != NULL)
        {
            ++count;
            if (hash(current->key, table.count) != i)
                printf("%d hashed to %ld, but was located at %ld\n",
                       current->key,
                       hash(current->key, table.count), i);
            current = current->next;
        }
    }
    if (count != ELEMENTS)
        std::cout << count << " elements found in hash table.  Should be " << ELEMENTS << std::endl;
    else
        std::cout << "All " << count << " elements found in hash table." << std::endl;
}

int main(void)
{
    unsigned int *buffer = (unsigned int *)big_random_block(SIZE);
    clock_t start, stop;
    start = clock();

    Table table;
    initialize_table(table, HASH_ENTRIES, ELEMENTS);

    for (int i = 0; i < ELEMENTS; i++)
    {
        add_to_table(table, buffer[i], (void *)NULL);
    }

    stop = clock();
    float elapsedTime = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Time to hash: " << elapsedTime << "ms" << std::endl;

    verify_table(table);

    free_table(table);
    free(buffer);

    return 0;
}