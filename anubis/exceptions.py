class ANUBISException(Exception):
    pass

def import_doc(f):
    """
    Copies the documentation of the function f into the decorated function
    """
    def func(g):
         g.__doc__ = f.__doc__
         return g
    return func
