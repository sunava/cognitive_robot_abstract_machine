from demos.thesis_new.demo_cut_all_breads_retry import main_cutting
from demos.thesis_new.demo_mix_all_bowls_retry import main_mixing
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker

if __name__ == "__main__":
    session = pycram_sessionmaker()()
    Base.metadata.create_all(session.bind)
    session.commit()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
    main_cutting()
    main_mixing()
