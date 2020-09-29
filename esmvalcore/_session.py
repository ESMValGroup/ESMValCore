from datetime import datetime

from .configuration import config
from .configuration._config_object import USER_CONFIG_DIR


class Session:
    """Importable session object.

    This class holds information about the current session. Different
    session directories can be accessed.

    Parameters
    ----------
    session_name : str
        Name of the session to initialize.
    """
    def __init__(self, session_name: str = 'session'):
        super().__init__()
        self.init_session_dir(session_name)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self._session_dir}')"

    def __str__(self):
        return f"{self.__class__.__name__}('{self._session_dir}')"

    def init_session_dir(self, name: str = 'session'):
        """Initialize session.

        The `name` is used to name the working directory, e.g.
        `recipe_example_20200916/`. If no name is given, such as in an
        interactive session, defaults to `session`.
        """
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        session_name = f"{name}_{now}"
        self._session_dir = config['output_dir'] / session_name

    @property
    def session_dir(self):
        return self._session_dir

    @property
    def preproc_dir(self):
        return self.session_dir / 'preproc'

    @property
    def work_dir(self):
        return self.session_dir / 'work'

    @property
    def plot_dir(self):
        return self.session_dir / 'plots'

    @property
    def run_dir(self):
        return self.session_dir / 'run'

    @property
    def config_dir(self):
        return USER_CONFIG_DIR


# must be initialized after config
session = Session()
