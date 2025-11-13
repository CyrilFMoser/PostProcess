from sklearn.neighbors import KDTree


def getTree(data) -> KDTree:
    coord = data["coord"]
    tree = KDTree(coord)
    return tree
