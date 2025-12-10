import warnings

# Silencia warnings do Ray (Deprecation, FutureWarning etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Alguns warnings do Ray vêm como UserWarning
warnings.filterwarnings("ignore", category=UserWarning)