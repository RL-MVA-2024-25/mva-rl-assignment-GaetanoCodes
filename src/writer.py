"""writer file"""

from termcolor import colored


class Writer:
    """writer"""

    def __init__(self):
        self.styles = {
            "t": {
                "color": "blue",
                "prefix": "\n###",
                "suffix": "###\n",
                "bold": True,
                "encadre": True,
            },
            "st": {
                "color": "light_grey",
                "prefix": "- ",
                "suffix": "",
                "bold": False,
                "encadre": False,
            },
            "txt": {
                "color": "white",
                "prefix": "",
                "suffix": "",
                "bold": False,
                "encadre": False,
            },
            "w": {
                "color": "yellow",
                "prefix": "⚠️ ",
                "suffix": "",
                "bold": False,
                "encadre": False,
            },
            "e": {
                "color": "red",
                "prefix": "❌ ",
                "suffix": "",
                "bold": False,
                "encadre": False,
            },
            "sc": {
                "color": "green",
                "prefix": "(*) ",
                "suffix": "",
                "bold": True,
                "encadre": False,
            },
        }

    def display(self, message: str, importance: str):
        """display"""
        if importance not in self.styles:
            raise ValueError(
                f"Importance '{importance}' inconnue. Choisissez parmi : {', '.join(self.styles.keys())}."
            )

        style = self.styles[importance]
        color = style["color"]
        prefix = style["prefix"]
        suffix = style["suffix"]
        bold = style["bold"]
        encadre = style["encadre"]

        # Formattage
        if encadre:
            formatted_message = f"{prefix}\n{message}\n{suffix}"
        else:
            formatted_message = f"{prefix}{message}{suffix}"

        # Appliquer le style gras si nécessaire
        if bold:
            formatted_message = f"\033[1m{formatted_message}\033[0m"

        # Affichage coloré
        print(colored(formatted_message, color))


if __name__ == "__main__":
    # Exemple d'utilisation
    writer = Writer()
    writer.display("Ceci est un titre", "t")
    writer.display("Ceci est un sous-titre", "st")
    writer.display("Ceci est un texte normal", "txt")
    writer.display("Ceci est un avertissement", "w")
    writer.display("Ceci est une erreur", "e")
