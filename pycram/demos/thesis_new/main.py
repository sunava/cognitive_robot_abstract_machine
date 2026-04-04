from demos.thesis_new import run_thesis_demo
from krrood.ormatic.utils import drop_database
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker

# pr2,hsrb,stretch,tiago,g1,justin,armar7
# apartment,kitchen,isr?, suturo, robocup, isr-testbed
if __name__ == "__main__":
    session = pycram_sessionmaker()()
    drop_database(session.bind)
    Base.metadata.create_all(session.bind)
    session.commit()

    run_thesis_demo(
        "cut",
        robot_name="pr2",
        environment_name="apartment",
    )
    run_thesis_demo(
        "cut",
        robot_name="pr2",
        environment_name="kitchen",
    )
    run_thesis_demo(
        "cut",
        robot_name="pr2",
        environment_name="isr-testbed",
    )

    run_thesis_demo("cut", robot_name="pr2")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
    run_thesis_demo("cut")
    run_thesis_demo("mix")
