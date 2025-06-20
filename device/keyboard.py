import numpy as np
from pynput.keyboard import Key, Listener


class Keyboard:

    def __init__(self, ):

        self._display_controls()
        self.start = False
        self.done = False
        self.finish = False
        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("s", "start a demo")
        print_command("d", "finish a demo")
        print_command("q", "quit")
        print("")

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            pass

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            if key.char == "d":
                self.done = True
            elif key.char == "s":
                self.start = True
            elif key.char == "q":
                self.finish = True

        except AttributeError as e:
            pass


class RecordKeyboard:

    def __init__(self, ):

        self._display_controls()
        self.start = False
        self.finish = False
        self.discard = False
        self.quit = False
        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("s", "start a demo")
        print_command("f", "finish a demo")
        print_command("d", "discard a demo")
        print_command("q", "quit")
        print("")

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            pass

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            if key.char == "f":
                self.finish = True
            elif key.char == "d":
                self.discard = True
            elif key.char == "s":
                self.start = True
                print('start!')
            elif key.char == "q":
                self.quit = True
        except AttributeError as e:
            pass


class TCPControl:

    def __init__(self, ):
        self.eps = 0.05

        self._display_controls()
        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("q-e", "change action frame")
        print_command("enter", "finish interaction")
        print("")

    def init(self, num_frame):
        self.pos = np.zeros(3)
        self.frame = 0
        self.num_frame = num_frame
        self.done = False

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] -= self.eps  # dec x
            elif key.char == "s":
                self.pos[0] += self.eps  # inc x
            elif key.char == "a":
                self.pos[1] -= self.eps  # dec y
            elif key.char == "d":
                self.pos[1] += self.eps  # inc y
            elif key.char == "f":
                self.pos[2] -= self.eps  # dec z
            elif key.char == "r":
                self.pos[2] += self.eps  # inc z

            elif key.char == "q":
                self.frame = max(self.frame-1, 0)
            elif key.char == "e":
                self.frame = min(self.frame+1, self.num_frame)

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            if key == Key.enter:
                self.done = True

        except AttributeError as e:
            pass


class KeyboardCounter:

    def __init__(self, ):

        self._display_controls()
        self.counter = {}
        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(
            on_press=self.on_press, on_release=self.on_release)
        # start listening
        self.listener.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        print("count any key")

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """

        try:
            pass

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """

        try:
            v = self.counter.get(key, 0)
            self.counter[key] = v + 1
        except AttributeError as e:
            pass

    def get(self, ):
        ret = self.counter
        self.counter = {}
        return ret


if __name__ == '__main__':
    import time
    device = Keyboard()
    while True:
        print(device.done)
        time.sleep(1)
