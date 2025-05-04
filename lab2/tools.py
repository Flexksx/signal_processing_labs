def generate_random_numbers(n):
    """
    Generate a list of n random numbers.
    """
    import random

    return [random.randint(1, 100) for _ in range(n)]
