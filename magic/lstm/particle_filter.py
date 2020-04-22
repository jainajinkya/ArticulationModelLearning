import copy

from filterpy.monte_carlo import systematic_resample
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import pickle
import os

## Utils
def sample_uniform_from_sphere(n_particles=100, r=1.):
    # Following: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    pts = np.random.rand(n_particles, 3) * (2) - 1.  # Centering for the complete circle
    e = np.random.exponential(0.5, n_particles)
    norms = np.sqrt(e + np.linalg.norm(pts, axis=1) ** 2)
    pts /= norms[:, np.newaxis]
    pts *= r  # scaling
    return pts


def to_plucker_coords(slope, pt):
    l = slope / np.linalg.norm(slope)
    return l, np.cross(pt, l)


def distance_bw_plucker_lines(line1, line2):
    # Based on formula from Plücker Coordinates for Lines in the Space by Prof. Yan-bin Jia
    # Verified by https://keisan.casio.com/exec/system/1223531414
    l1, m1 = line1[:3].astype(np.float), line1[3:6].astype(np.float)
    l2, m2 = line2[:3].astype(np.float), line2[3:6].astype(np.float)
    norm_cross_prod = np.linalg.norm(np.cross(l1, l2))
    if norm_cross_prod == 0:
        s = np.linalg.norm(l2) / np.linalg.norm(l1)
        dist = np.linalg.norm(np.cross(l1, (m1 - m2 / s))) / np.linalg.norm(l1) ** 2
    else:
        dist = abs(np.dot(l1, m2) + np.dot(l2, m1)) / norm_cross_prod
    return dist


def construct_line_from_plucker(l, m, lims=(-0.1, 0.1)):
    l /= np.linalg.norm(l)   # Normalize
    m -= (np.dot(l, m)/np.dot(l, l)) * l  # orthogonalize
    pt_prep = np.cross(l, m)
    if abs(l[0]) < 1e-9:
        x = np.array([pt_prep[0]]*10)
        if abs(l[1]) < 1e-9:
            y = np.array([pt_prep[1]]*10)
            z = np.linspace(lims[0], lims[1], num=10)
        else:
            y = np.linspace(lims[0], lims[1], num=10)
            z = pt_prep[2] + (l[2]/l[1])*(y - pt_prep[1])
    else:
        x = np.linspace(lims[0], lims[1], num=10)
        y = pt_prep[1] + (l[1]/l[0])*(x - pt_prep[0])
        z = pt_prep[2] + (l[2]/l[0])*(x - pt_prep[0])
    return x, y, z


## Plotting Utils
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


## Main functions
def create_initial_particles(n_particles, x_init=None, type=None):
    if type == 'uniform':
        # Sample l-hat
        l_hat = np.random.rand(n_particles, 3)
        l_hat /= np.linalg.norm(l_hat, axis=1)[:, np.newaxis]
        # Sample the point through which the line passes
        p = sample_uniform_from_sphere(n_particles, r=1)  # Considering a sphere of radius 1 m
    else:
        l_init, m_init = x_init[:3], x_init[3:6]
        l_init /= np.linalg.norm(l_init)  # Normalize

        # Sample l_hat
        l_hat = l_init + np.random.multivariate_normal(np.zeros(3), 5e-6*np.eye(3), n_particles)
        l_hat /= np.linalg.norm(l_hat, axis=1)[:, np.newaxis]

        # Point through which initial line passes
        p_prep = np.cross(l_init, m_init)
        p = p_prep + sample_uniform_from_sphere(n_particles, r=1e-3)  # Considering a sphere of radius 1 cm

    # Moment vector
    m = np.cross(p, l_hat)  # May need to add some particle for the drawer axis
    return np.concatenate((l_hat, m), axis=1)


def predict(particles):
    # Assuming Identity forward propagation
    return particles


def update(particles, z=None, R=1e-3):
    weights = np.zeros(np.shape(particles)[0]) + 1.e-300  # avoid round-off to zero
    for i, x in enumerate(particles):
        # Calculate perpendicular distance
        dist = distance_bw_plucker_lines(x, z)

        # Calculate angular difference
        ang = np.arccos(np.dot(x[:3], z[:3]) / (np.linalg.norm(x[:3]) * np.linalg.norm(z[:3])))

        # Calculate wts
        wts_dist = scipy.stats.norm(0, R).pdf(dist)
        wts_ang = scipy.stats.norm(0, R).pdf(ang)
        weights[i] = wts_ang + wts_dist

    weights /= sum(weights)  # normalize
    return weights


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    print("RESAMPLING")
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    """ This is wrong. We need to have better probabilistic distribution of these values and not just mean"""
    l_hat = particles[:, :3]
    m = particles[:, 3:6]
    pts = np.cross(l_hat, m)

    mean_l_hat = np.average(l_hat, weights=weights, axis=0)
    var_l_hat = np.average((l_hat - mean_l_hat) ** 2, weights=weights, axis=0)

    # Construct pts on line
    mean_pt = np.average(pts, weights=weights, axis=0)
    mean_m = np.cross(mean_pt, mean_l_hat)
    # mean_m = np.average(m, weights=weights, axis=0)
    var_m = np.average((m - mean_m) ** 2, weights=weights, axis=0)
    return mean_l_hat, var_l_hat, mean_m, var_m


def gs_orthonormalize(X):
    X1 = X[:, :3]
    X2 = X[:, 3:6]

    def gs_cofficient(v1, v2):
        return np.dot(v2, v1) / np.dot(v1, v1)

    def multiply(cofficient, v):
        return cofficient * v

    def proj(v1, v2):
        # projecting v2 on v1
        return multiply(gs_cofficient(v1, v2), v1)

    # Normalize 1st vector row-wise
    Y1 = copy.copy(X1 / np.linalg.norm(X1, axis=1)[:, np.newaxis])
    Y2 = []

    for v1, v2 in zip(X1, X2):
        y = v2 - proj(v1, v2)
        # print("Dot product b/w v1 and y2: {}".format(np.dot(v1, y)))          # Sanity check
        Y2.append(y)
    return np.concatenate((Y1, np.array(Y2)), axis=1)


def run_pf1(n_particles, observations, sensor_std_err=1e-3):
    # create particles and weights
    particles = create_initial_particles(n_particles, x_init=observations[0, :])
    initial_particles = copy.copy(particles)

    l_means = []
    l_vars = []
    m_means = []
    m_vars = []

    for zs in observations[0:, :]:
        # Propagate particles forward
        predict(particles)

        # incorporate measurements
        weights = update(particles, z=zs, R=sensor_std_err)

        # resample if too few effective particles
        if neff(weights) < n_particles / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / n_particles)

        mean_l_hat, var_l_hat, mean_m, var_m = estimate(particles, weights)
        l_means.append(mean_l_hat)
        l_vars.append(var_l_hat)
        m_means.append(mean_m)
        m_vars.append(var_m)

        # print("Particles: ", particles)
        # print("\n weights: ", weights)
        # input("Press enter to continue")

    return np.array(l_means), np.array(l_vars), np.array(m_means), np.array(m_vars), initial_particles


if __name__ == "__main__":
    # from numpy.random import seed
    # seed(2)

    # Load prediction data
    f_name = os.path.expanduser("~/research/ArticulationModel/results/lstm_all_2/test_prediction_data.pkl")
    data = pickle.load(open(f_name, 'rb'))
    all_observations = data['predictions']
    all_labels = data['labels']

    # Perform gram-Schmidt orthonormalization on observations
    obj_idx = 98
    obs = gs_orthonormalize(all_observations[obj_idx][:, :6])

    # Run filter
    l_hat_means, l_hat_vars, m_means, m_vars, init_particles = run_pf1(n_particles=1000,
                                                                       observations=obs, sensor_std_err=0.1)

    # print("Prior l_hat err: {}".format(np.mean(all_labels[obj_idx][0, :3] - all_observations[obj_idx][:, :3], axis=0)))
    # print("Post l_hat err: {}".format(all_labels[obj_idx][0, :3] - l_hat_means[-1, :]))
    #
    # print("Prior m err: {}".format(np.mean(all_labels[obj_idx][0, 3:6] - all_observations[obj_idx][:, 3:6], axis=0)))
    # print("Post m err: {}".format(all_labels[obj_idx][0, 3:6] - m_means[-1, :]))

    dists = []
    angs = []
    for pred in all_observations[obj_idx][:, :6]:
        dists.append(distance_bw_plucker_lines(all_labels[obj_idx][0, :6], pred))
        angs.append(np.arccos(np.dot(all_labels[obj_idx][0, :3], pred[:3]) /
                              (np.linalg.norm(all_labels[obj_idx][0, :3]) * np.linalg.norm(pred[:3]))))

    print("Prior mean angular error: {}".format(np.mean(np.array(angs), axis=0)))
    print("Post angular error: {}".format(np.arccos(np.dot(all_labels[obj_idx][0, :3], l_hat_means[-1, :]) /
                              (np.linalg.norm(all_labels[obj_idx][0, :3]) * np.linalg.norm(l_hat_means[-1, :])))))

    print("Prior mean distance error: {}".format(np.mean(np.array(dists), axis=0)))
    print("Post distance error: {}".format(
        distance_bw_plucker_lines(all_labels[obj_idx][0, :6], np.concatenate((l_hat_means[-1, :], m_means[-1, :])))))


    ''' Plots '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # 3D Line plots
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    x_label, y_label, z_label = construct_line_from_plucker(all_labels[obj_idx][0, :3], all_labels[obj_idx][0, 3:6],
                                                            lims=(-100, 100))
    ax.plot3D(x_label, y_label, z_label, 'b', label='GT-Label', linewidth=2)

    x_pred, y_pred, z_pred = [], [], []
    for i, pt in enumerate(all_observations[obj_idx]):
        x_p, y_p, z_p = construct_line_from_plucker(pt[:3], pt[3:6])
        ax.plot3D(x_p, y_p, z_p, 'r')
        x_pred.append(x_p)
        y_pred.append(y_p)
        z_pred.append(z_p)

    ax.plot3D(x_p, y_p, z_p, 'r', label='Predictions')  # For generating labels

    x_pf, y_pf, z_pf = construct_line_from_plucker(l_hat_means[-1, :], m_means[-1, :])
    ax.plot3D(x_pf, y_pf, z_pf, 'k', label='PF Prediction')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim3d([-1., 1.])
    ax.set_ylim3d([-1., 1.])
    ax.set_zlim3d([-1., 1.])
    ax.legend()
    # set_axes_equal(ax)

    # Projected graphs
    figs, axs = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.3)

    # XY PLot
    axs[0].plot(x_label, y_label, 'bo', linewidth=10, markersize=6)
    for x_p, y_p in zip(x_pred, y_pred):
        axs[0].plot(x_p, y_p, 'r')
    axs[0].plot(x_pf, y_pf, 'k', linewidth=3)

    axs[0].set_title('X-Y Plot')
    axs[0].set_xlabel('X axis')
    axs[0].set_ylabel('Y axis', labelpad=0)
    axs[0].set_xlim([-0.25, 0.25])


    # XZ PLot
    axs[1].plot(x_label, z_label, 'b', label='GT-Label')
    for x_p, z_p in zip(x_pred, z_pred):
        axs[1].plot(x_p, z_p, 'r')
    axs[1].plot(x_p, z_p, 'r', label='Predictions')  # For generating labels
    axs[1].plot(x_pf, z_pf, 'k', label='PF Predictions', linewidth=3)

    axs[1].legend(loc="upper right")
    axs[1].set_title('X-Z Plot')
    axs[1].set_xlabel('X axis')
    axs[1].set_ylabel('Z axis', labelpad=-20)
    axs[1].set_xlim([-0.25, 0.25])

    # YZ PLot
    axs[2].plot(y_label, z_label, 'b')
    for y_p, z_p in zip(y_pred, z_pred):
        axs[2].plot(y_p, z_p, 'r')
    axs[2].plot(y_pf, z_pf, 'k', linewidth=3)

    axs[2].set_title('Y-Z Plot')
    axs[2].set_xlabel('Y axis')
    axs[2].set_ylabel('Z axis', labelpad=-20)
    axs[2].set_xlim([-0.5, 0.])

    # ## Debugging nitial Particles
    # figs1, axs1 = plt.subplots(1, 3)
    # plt.subplots_adjust(wspace=0.3)
    #
    # x_p_init, y_p_init, z_p_init = [], [], []
    # for i, pt in enumerate(init_particles):
    #     x_p, y_p, z_p = construct_line_from_plucker(pt[:3], pt[3:6])
    #     x_p_init.append(x_p)
    #     y_p_init.append(y_p)
    #     z_p_init.append(z_p)
    #
    # x_pred = np.array(x_pred)
    # y_pred = np.array(y_pred)
    # z_pred = np.array(z_pred)
    #
    # # XY PLot
    # axs1[0].plot(x_pred[0, :], y_pred[0, :], 'r', linewidth=5)
    # for x_p, y_p in zip(x_p_init, y_p_init):
    #     axs1[0].plot(x_p, y_p, 'g')
    #
    # axs1[0].set_title('X-Y Plot')
    # axs1[0].set_xlabel('X axis')
    # axs1[0].set_ylabel('Y axis', labelpad=0)
    # axs1[0].set_xlim([-0.5, 0.5])
    #
    #
    # # XZ PLot
    # axs1[1].plot(x_pred[0, :], z_pred[0, :], 'r', label='1st observation', linewidth=5)
    # for x_p, z_p in zip(x_p_init, z_p_init):
    #     axs1[1].plot(x_p, z_p, 'g')
    # axs1[1].plot(x_p, z_p, 'g', label='PF init')  # For generating labels
    #
    # axs1[1].legend(loc="upper right")
    # axs1[1].set_title('X-Z Plot')
    # axs1[1].set_xlabel('X axis')
    # axs1[1].set_ylabel('Z axis', labelpad=-20)
    # axs1[1].set_xlim([-0.5, 0.5])
    #
    # # YZ PLot
    # axs1[2].plot(y_pred[0, :], z_pred[0, :], 'r', linewidth=5)
    # for y_p, z_p in zip(y_p_init, z_p_init):
    #     axs1[2].plot(y_p, z_p, 'g')
    #
    # axs1[2].set_title('Y-Z Plot')
    # axs1[2].set_xlabel('Y axis')
    # axs1[2].set_ylabel('Z axis', labelpad=-20)
    # axs1[2].set_xlim([-0.5, 0.5])

    plt.show()



