from pathlib import Path
from pprint import pformat


class OutputItem:
    """Base container for recipe output.

    Use `OutputItem.create(path='<filename>')` to initialize a suitable
    subclass.
    """
    def __init__(self, filename, data={}):
        self.data = data
        self.filename = Path(filename)

    def __repr__(self):
        return f'{self.__class__.__name__}(filename={self.filename.name!r},\ndata={pformat(self.data)})'

    def __getattr__(self, item):
        if item in self.data:
            return self.data[item]
        else:
            raise AttributeError(
                f'{self.__class__.__name__!r} has no attribute {item}')

    @classmethod
    def create(cls, filename, data={}):
        ext = Path(filename).suffix
        if ext in ('.png', ):
            item_class = ImageItem
        elif ext in ('.nc', ):
            item_class = DataItem
        else:
            item_class = cls

        return item_class(filename=filename, data=data)


class ImageItem(OutputItem):
    """Container for image output."""
    def _repr_png_(self):
        """Render png as image in Jupyter Notebook."""
        from IPython.display import Image
        return Image(self.filename).data


class DataItem(OutputItem):
    """Container for image output."""
    def load_xarray(self):
        """Load data using xarray."""
        import xarray as xr
        return xr.load_dataset(self.filename)

    def load_iris(self):
        """Load data using iris."""
        import iris
        return iris.load(str(self.filename))
