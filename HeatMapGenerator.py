import numpy
import tensorflow as tf
import matplotlib.pyplot as plt


#
#
#
def gaussian_kernel(image_size: int, sigma: float, normalize_kernel: bool):
    ax = numpy.arange(-image_size // 2 + 1., image_size // 2 + 1.)
    xx, yy = numpy.meshgrid(ax, ax)
    kernel = numpy.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    if normalize_kernel:
        return kernel / numpy.sum(kernel)
    return kernel


#
#
#
def build_heatmap(locations: tf.Tensor, map_width: int, map_height: int, sigma: float, normalize_kernel: bool=False):
    """
        Build a heatmap from a variable length list of locations in tensorflow

        Parameters
            locations           int32 tensor [num_locations, 2]
            map_width           int
            map_height          int
            sigma               float
            normalize_kernel    bool

        Return:
            heatmap as float32 tensor

        Example:
            locations = tf.placeholder(shape=(None, 2), dtype=tf.int32)
            locations[i, 0] = x position of the i-th Gaussian
            locations[i, 1] = y position of the i-th Gaussian

    """
    num_joints = tf.shape(locations)[0]
    heatmap = tf.Variable(numpy.zeros([map_height, map_width]), name='heatmap', dtype=tf.float32, trainable=False)

    #
    # Build kernel
    #
    kernel_size = 21
    kernel_offset = int((kernel_size - 1) / 2)
    g_kernel = gaussian_kernel(kernel_size, sigma, normalize_kernel)
    g_kernel = tf.constant(g_kernel, name='gaussian_kernel', dtype=tf.float32)
    kernel_offset = tf.constant(kernel_offset, name='kernel_offset', dtype=tf.int32)
    map_width_ = tf.constant(map_width - 1, name='map_width', dtype=tf.int32)
    map_height_ = tf.constant(map_height - 1, name='map_height', dtype=tf.int32)

    #
    # Translate Kernel
    #
    index = tf.constant(0)

    def cond(idx, _):
        return tf.less(idx, num_joints)

    def operation(idx, hmap):
        pad_h = tf.maximum(map_height_ - locations[idx, 1] - kernel_offset, 0)
        pad_w = tf.maximum(map_width_ - locations[idx, 0] - kernel_offset, 0)
        st_h = locations[idx, 1] - kernel_offset
        st_w = locations[idx, 0] - kernel_offset
        tmp = tf.pad(g_kernel, ((tf.maximum(st_h, 0), pad_h), (tf.maximum(st_w, 0), pad_w)), mode='CONSTANT', constant_values=0)
        hmap = tf.add(hmap, tmp[-tf.minimum(st_h, 0):map_height - tf.minimum(st_h, 0), -tf.minimum(st_w, 0):map_width - tf.minimum(st_w, 0)])
        return [idx + 1, hmap]

    _, heatmap = tf.while_loop(cond, operation, [index, heatmap])

    return heatmap


#
#
#
sess = tf.Session()

#
# Init
#
joint_list_ = tf.placeholder(shape=(None, 2), name='joint_list', dtype=tf.int32)
heatmap_ = build_heatmap(joint_list_, 100, 200, 2.5, normalize_kernel=False)

#
# Example
#
sess.run(tf.global_variables_initializer())
ret = sess.run(heatmap_, feed_dict={joint_list_: numpy.array([[-1, 20], [1, 30], [2, 199]])})
print(ret.shape)
plt.imshow(ret, interpolation='none')
plt.show()
ret = sess.run(heatmap_, feed_dict={joint_list_: numpy.array([[20, 20], [30, 20], [40, 20], [60, 20]])})
print(ret.shape)
plt.imshow(ret, interpolation='none')
plt.show()
sess.close()

