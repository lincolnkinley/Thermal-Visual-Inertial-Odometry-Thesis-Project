import numpy as np
import gtsam
import matplotlib.pyplot as plt
from gtsam.utils import plot

def vector3(x, y, z):
    """Create 3d double numpy array."""
    return np.array([x, y, z], dtype=float)

def Process_g2o(input_file, output_file):
    graph, initial = gtsam.readG2o(input_file, False)
    priorModel = gtsam.noiseModel.Diagonal.Variances(vector3(1e-6, 1e-6, 1e-8))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(), priorModel))

    params = gtsam.GaussNewtonParams()
    params.setVerbosity("Termination")
    params.setMaxIterations(100)

    optimizer = gtsam.GaussNewtonOptimizer(graph, initial, params)
    result = optimizer.optimize()

    print("Optimization complete")
    print("initial error = ", graph.error(initial))
    print("final error = ", graph.error(result))

    print("Writing results to file: ", output_file)
    graphNoKernel, _ = gtsam.readG2o(input_file, False)
    gtsam.writeG2o(graphNoKernel, result, output_file)
    print("Done!")
    '''
    resultPoses = gtsam.utilities.extractPose2(result)
    for i in range(resultPoses.shape[0]):
        plot.plot_pose2(1, gtsam.Pose2(resultPoses[i, :]))
    plt.show()
    '''

if __name__ == '__main__':
    Process_g2o("6Images/output.g2o", "6Images/optimized.g2o")
