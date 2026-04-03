from textual.app import App
from orion.config.config import Config
from orion.session.session import SessionManager
from orion.cli.screens.splash import SplashScreen
from orion.cli.screens.setup import SetupScreen


class OrionApp(App):
    TITLE = "Orion CLI"
    CSS = """
    $accent: #8B1A1A;
    """

    def on_mount(self):
        self.config = Config.load()
        self.session_manager = SessionManager()
        if not self.config.is_configured:
            self.push_screen(SetupScreen())
        else:
            self.push_screen(SplashScreen())


def main():
    OrionApp().run()


if __name__ == "__main__":
    main()
