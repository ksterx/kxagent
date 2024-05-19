from dataclasses import dataclass, field


@dataclass
class SensorData:
    name: str
    data: dict = field(default_factory=dict)


@dataclass
class ActuatorData:
    name: str
    data: dict = field(default_factory=dict)


class Brain:
    def __init__(self):
        pass

    def receive(self, data):
        pass

    def send(self):
        pass


class Agent:
    def __init__(self, brain, sensors, actuators, id=None):
        # sensors can be a list of Sensor objects such as vision, audio, etc.
        self.brain = brain
        self.sensors = sensors
        self.actuators = actuators
        self.id = id

    def sense(self, preprocessed=False):
        for sensor in self.sensors:
            if preprocessed:
                sensor_data = sensor.read()
            else:
                sensor_data = sensor.process()

            self.brain.receive(sensor_data)

    def act(self):
        for actuator in self.actuators:
            actuator_data = actuator.process()
            self.brain.send(actuator_data)

    def run(self):
        while True:
            self.sense()
            self.act()


@dataclass
class CommandData:
    to: str
    action: ActuatorData


class Commander:
    def __init__(self, brain, agents):
        self.brain = brain
        self.agents = agents

    async def collect(self):
        pass

    async def command(self):
        pass
