# start by importing some things we will need
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import multivariate_normal
from math import trunc 


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution around [0,0]
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0, cov_0)
    belief = RV.pdf(pos)
    return belief


# Now let's define the predict function

def run_odometry(N_l, N_r, robot):
    R = robot['wheel_radius']
    L = robot['wheel_baseline']
    N = robot['encoder_resolution']
    
    d_l = R * N_l / N * 2 * np.pi
    d_r = R * N_r / N * 2 * np.pi
    d_A = (d_l + d_r) / 2
    delta_phi = (d_r - d_l) / L

    return d_A, delta_phi

def histogram_predict(belief, dt, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
    belief_in = belief
    delta_t = dt

    # TODO calculate v and w from ticks using kinematics. You will need  some parameters in the `robot_spec` defined above
    d_A, delta_phi = run_odometry(left_encoder_ticks, right_encoder_ticks, robot_spec)
    dx = d_A * np.cos(grid_spec['phi'])
    dy = d_A * np.sin(grid_spec['phi'])

    # TODO propagate each centroid forward using the kinematic function
    d_t = grid_spec['d'] + dy
    phi_t = grid_spec['phi'] + delta_phi

    p_belief = np.zeros(belief.shape)

    # Accumulate the mass for each cell as a result of the propagation step
    for i in range(belief.shape[0]):
        for j in range(belief.shape[1]):
            # If belief[i,j] there was no mass to move in the first place
            if belief[i, j] > 0:
                # Now check that the centroid of the cell wasn't propagated out of the allowable range
                if (
                    d_t[i, j] > grid_spec["d_max"]
                    or d_t[i, j] < grid_spec["d_min"]
                    or phi_t[i, j] < grid_spec["phi_min"]
                    or phi_t[i, j] > grid_spec["phi_max"]
                ):
                    continue

                # TODO Now find the cell where the new mass should be added
                i_new = trunc((d_t[i,j]-grid_spec['d_min'])/grid_spec['delta_d'])
                j_new = trunc((phi_t[i,j]-grid_spec['phi_min'])/grid_spec['delta_phi'])

                p_belief[i_new, j_new] += belief[i, j]

    # Finally we are going to add some "noise" according to the process model noise
    # This is implemented as a Gaussian blur
    s_belief = np.zeros(belief.shape)
    gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")

    if np.sum(s_belief) == 0:
        return belief_in
    belief = s_belief / np.sum(s_belief)
    return belief


# We will start by doing a little bit of processing on the segments to remove anything that is
# behing the robot (why would it be behind?) or a color not equal to yellow or white


def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments


def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])

    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec["linewidth_white"]
        else:  # left edge of white lane
            d_i = -d_i
            phi_i = -phi_i
        d_i -= road_spec["lanewidth"] / 2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec["linewidth_yellow"]
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec["lanewidth"] / 2 - d_i

    return d_i, phi_i


def generate_measurement_likelihood(segments, road_spec, grid_spec):
    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec["d"].shape)

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if (
            d_i > grid_spec["d_max"]
            or d_i < grid_spec["d_min"]
            or phi_i < grid_spec["phi_min"]
            or phi_i > grid_spec["phi_max"]
        ):
            continue

        # TODO find the cell index that corresponds to the measurement d_i, phi_i
        i = trunc((d_i-grid_spec['d_min'])/grid_spec['delta_d'])
        j = trunc((phi_i-grid_spec['phi_min'])/grid_spec['delta_phi'])

        # Add one vote to that cell
        measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    if measurement_likelihood is not None:
        # TODO: combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid
        # probability distribution

        # replace this with something that combines the belief and the measurement_likelihood
        belief = measurement_likelihood * belief
        tot = np.sum(belief)
        belief = belief / tot
    return measurement_likelihood, belief
