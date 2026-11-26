from importlib.resources import files

# Data files shipped inside the package, one constant per subdirectory.
GRAPHS = files(__name__) / "graphs"
