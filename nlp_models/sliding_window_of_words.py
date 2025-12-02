def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i ] + words[(i + 1): ( i + C + 1)]
        yield context_words, center_word
        i += 1


# Example usage:
if __name__ == "__main__":
    words = ["I", "love", "learning", "about", "geometric", "deep", "learning", "."]
    C = 2
    for context, center in get_windows(words, C):
        print(f"Context: {context}, Center: {center}")

    # another example using machine learning syntax style
    for x, y in get_windows(
        ["I", "love", "learning", "about", "geometric", "deep", "learning", "."], 
        2
    ):
        # print(f"Context: {x}, Center: {y}")
        print(f'{x}\t{y}')