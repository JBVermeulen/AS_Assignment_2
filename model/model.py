from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link
import pandas as pd
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Since there is only one road in the Demo, the paths are added with the road info;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    def __init__(
        self,
        seed=None,
        x_max=500,
        y_max=500,
        x_min=0,
        y_min=0,

        scenario=None,
        CatA=0,
        CatB=0,
        CatC=0,
        CatD=0
    ):

        super().__init__()
        if scenario is None:
            scenario = {'CatA': CatA, 'CatB': CatB, 'CatC': CatC, 'CatD': CatD}

        # Bridge destruction chances
        self.cat_a = scenario['CatA']
        self.cat_b = scenario['CatB']
        self.cat_c = scenario['CatC']
        self.cat_d = scenario['CatD']

        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []

        self.generate_model()

    # ---------------------------------------------------------------
    def generate_model(self):
        """
        generate the simulation model according to the csv file component information
        """

        data_path = Path(__file__).resolve().parents[1] / "data" / "test_data3.csv"
        df = pd.read_csv(data_path)

        roads = ['N1']

        df_objects_all = []
        for road in roads:

            df_objects_on_road = df[df['road'] == road].sort_values(by=['id'])

            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)

                path_ids = df_objects_on_road['id']
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids

                path_ids = path_ids[::-1]
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids

        df = pd.concat(df_objects_all)

        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():

                model_type = row['model_type']
                agent = None

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)

                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sinks.append(agent.unique_id)

                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)

                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], row['name'], row['road'], row['quality'])

                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], row['name'], row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    # ---------------------------------------------------------------
    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return self.path_ids_dict[source, sink]
    
    # ---------------------------------------------------------------
    def record_completed_trip(self, vehicle):
        """
        Record the completed trip of a vehicle when it arrives at the sink
        """
        print(f"Vehicle {vehicle.unique_id} completed its trip. Total delay time: {vehicle.delay_time} minutes.")

    # ---------------------------------------------------------------
    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()


# EOF -----------------------------------------------------------