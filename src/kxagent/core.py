import uuid
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class SensorData:
    name: str
    data: dict = field(default_factory=dict)


@dataclass
class ActuatorData:
    name: str
    data: dict = field(default_factory=dict)


class Brain:
    def __init__(self, model):
        self.model = model
        self.buffer = []

    def receive(self, data):
        pass

    def send(self):
        pass


class Worker:
    def __init__(self):
        self.uuid = uuid.uuid4()
        self.status = "waiting"  # waiting/processing
        self.buffer = []

    def process(self, data):
        self.status = "processing"
        # process data
        self.send(data)

    def send(self, worker):
        if worker.status == "waiting":
            return
        else:
            raise NotImplementedError


class Interpreter(Worker):
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def process(self):
        return SensorData(self.name, self.data)

    def read(self):
        return self.data


class Actor(Worker):
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def process(self):
        return ActuatorData(self.name, self.data)


class Agent:
    def __init__(
        self,
        brain: Brain,
        interpreters: list[Interpreter],
        actors: list[Actor],
        unique_id: str | None = None,
        idx: str | None = None,
        name: str | None = None,
    ):
        # sensors can be a list of Sensor objects such as vision, audio, etc.
        self.brain = brain
        self.interpreters = interpreters
        self.actors = actors
        self.uuid = unique_id
        self.idx = idx
        self.name = name
        self.status = "waiting"  # waiting/processing

    def __eq__(self, other):
        return self.uuid == other.uuid

    def sense(self, preprocessed=False):
        for sensor in self.interpreters:
            if preprocessed:
                sensor_data = sensor.read()
            else:
                sensor_data = sensor.process()

            self.brain.receive(sensor_data)

    def act(self):
        for actor in self.actors:
            actuator_data = actor.process()
            self.brain.send(actuator_data)

    def run(self):
        while True:
            self.sense()
            self.act()


@dataclass
class Connection:
    agent_from: Agent
    agent_to: Agent
    uuid: str = uuid.uuid4()

    def has_agent(self, agent: Agent):
        return self.agent_from == agent or self.agent_to == agent


class Graph:
    def __init__(self, agents: list[Agent], connections: list[Connection]):
        self.agents = agents
        for i, agent in enumerate(agents):
            agent.idx = i

        self.connections = connections
        self.matrix = np.zeros((len(agents), len(agents)))
        for c in connections:
            self.matrix[c.agent_from.idx, c.agent_to.idx] = 1

    def add_agent(self, agent, connections: list[Connection] | None = None):
        if not connections:
            raise ValueError(
                "Any connection provided. Agent must be connected to at least one other agent."
            )

        self.agents.append(agent)
        self.matrix = np.zeros((len(self.agents), len(self.agents)))
        for c in connections:
            if not c.has_agent(agent):
                raise ValueError("Agent not in connection")
            self.update(c.agent_from, c.agent_to)

    def update(self, agent_from: Agent, agent_to: Agent):
        self.matrix[agent_from.idx, agent_to.idx] = 1

    def update_connections(self, connections: list[Connection]):
        for c in connections:
            self.update(c.agent_from, c.agent_to)

    def remove_agent(self, agent: Agent):
        self.agents.remove(agent)
        for i, a in enumerate(self.agents):
            a.idx = i
        self.matrix = np.zeros((len(self.agents), len(self.agents)))

        new_connections = []
        for c in self.connections:
            if not c.has_agent(agent):
                self.update(c.agent_from, c.agent_to)
                new_connections.append(c)

        self.connections = new_connections


class Group:
    def __init__(self, graph):
        self.graph = graph

    def run(self):
        for agent in self.graph.agents:
            agent.run()
