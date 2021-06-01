# This is a separate file to make model export work

# Files starting with upper-case character are pictures of a cat
def label_func(f):
    return "cat" if f[0].isupper() else "dog"