from pathlib import Path

from jass.features.labels_action_full import LabelSetActionFull

from jass_mu_zero.mu_zero.metrics.vpkl import VPKL
from test.util import get_test_config


def test_vpkl_0_step():
    config = get_test_config()

    testee = VPKL(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=0
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 1

    del testee

def test_vpkl_1_step():
    config = get_test_config()

    testee = VPKL(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=1
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 2

    del testee


def test_vpkl_30_step():
    config = get_test_config()

    testee = VPKL(
        samples_per_calculation=2,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=30
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 31

    del testee


def test_vpkl_larger_batch():
    config = get_test_config()

    testee = VPKL(
        samples_per_calculation=128,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path=str(Path(__file__).parent.parent / "resources" / "imperfect_resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 17

    del testee


def test_vpkl_more_steps_larger_batch_perfect():
    config = get_test_config(cheating=True)

    testee = VPKL(
        samples_per_calculation=32,
        label_length=LabelSetActionFull.LABEL_LENGTH,
        worker_config=config,
        network_path = str(Path(__file__).parent.parent / "resources" / "perfect_resnet_random.pd"),
        n_steps_ahead=16
    )

    testee.poll_till_next_result_available()

    result = testee.get_latest_result()

    assert len(result) == 17

    del testee