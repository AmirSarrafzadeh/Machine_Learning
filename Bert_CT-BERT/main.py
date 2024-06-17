MOD = 10 ** 9 + 7


def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]


def union(parent, rank, x, y):
    rootX = find(parent, x)
    rootY = find(parent, y)

    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1


def count_ways(n, m, roads):
    parent = list(range(n))
    rank = [0] * n

    for u, v in roads:
        union(parent, rank, u - 1, v - 1)

    component_size = [0] * n
    for i in range(n):
        root = find(parent, i)
        component_size[root] += 1

    components = [size for size in component_size if size > 0]

    total_ways = pow(2, n, MOD)
    invalid_ways = 0

    for size in components:
        if size < n:
            invalid_ways = (invalid_ways + pow(2, size, MOD)) % MOD

    result = (total_ways - invalid_ways + MOD) % MOD
    return result


# Input
n, m = map(int, input().split())
roads = [tuple(map(int, input().split())) for _ in range(m)]

# Output the result
print(count_ways(n, m, roads))
