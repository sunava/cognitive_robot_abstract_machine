from demos.thesis_new import run_thesis_demo
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker

if __name__ == "__main__":
    session = pycram_sessionmaker()()
    Base.metadata.create_all(session.bind)
    session.commit()
    run_thesis_demo("wipe", robot_name="tiago")
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
