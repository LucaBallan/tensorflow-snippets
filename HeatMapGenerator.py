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
def build_heatmap_sprite(locations: tf.Tensor, map_width: int, map_height: int, sigma: float, normalize_kernel: bool=False):
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
    num_locations = tf.shape(locations)[0]
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
        return tf.less(idx, num_locations)

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
def build_heatmap_conv(locations: tf.Tensor, map_width: int, map_height: int, sigma: float, normalize_kernel: bool=False):
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
    #
    # Build kernel
    #
    kernel_size = 21
    g_kernel = gaussian_kernel(kernel_size, sigma, normalize_kernel)
    g_kernel = numpy.reshape(g_kernel, [kernel_size, kernel_size, 1, 1])
    g_kernel = tf.constant(g_kernel, name='gaussian_kernel', dtype=tf.float32)

    #
    #
    #
    offset = tf.constant(int((kernel_size - 1) / 2), dtype=tf.int32)
    map_wh = tf.constant([map_width + kernel_size, map_height + kernel_size], name='map_width', dtype=tf.int32)
    locations = tf.add(locations, offset)
    out_of_border = tf.logical_and(tf.less(locations, map_wh), tf.greater_equal(locations, tf.zeros([1, 2], dtype=tf.int32)))
    out_of_border = tf.reduce_all(out_of_border, axis=1)
    locations = tf.boolean_mask(locations, out_of_border)

    #
    #
    #
    num_locations = tf.shape(locations)[0]
    heatmap = tf.scatter_nd(locations, tf.ones([num_locations], dtype=tf.float32), [map_width + kernel_size, map_height + kernel_size])
    heatmap = tf.reshape(tf.transpose(heatmap), [1, map_height + kernel_size, map_width + kernel_size, 1])
    heatmap = tf.nn.conv2d(heatmap, g_kernel, strides=[1, 1, 1, 1], padding='SAME')
    heatmap = tf.reshape(heatmap, [map_height + kernel_size, map_width + kernel_size])

    return heatmap[offset:(map_height + offset), offset:(map_width + offset)]


#
#
#
sess = tf.Session()

#
# Init
#
joint_list_ = tf.placeholder(shape=(None, 2), name='joint_list', dtype=tf.int32)
heatmap_sprite = build_heatmap_sprite(joint_list_, 100, 200, 2.5, normalize_kernel=False)
heatmap_conv = build_heatmap_conv(joint_list_, 100, 200, 2.5, normalize_kernel=False)

#
# Example
#
sess.run(tf.global_variables_initializer())
ret1, ret2 = sess.run([heatmap_sprite, heatmap_conv], feed_dict={joint_list_: numpy.array([[-1, 20], [1, 30], [2, 199]])})
print(ret1.shape)
print(ret2.shape)
plt.figure()
plt.imshow(ret1, interpolation='none')
plt.title('heatmap_sprite')
plt.figure()
plt.imshow(ret2, interpolation='none')
plt.title('heatmap_conv')
plt.show()
ret1, ret2 = sess.run([heatmap_sprite, heatmap_conv], feed_dict={joint_list_: numpy.array([[20, 20], [30, 20], [40, 20], [60, 20]])})
print(ret1.shape)
print(ret2.shape)
plt.figure()
plt.imshow(ret1, interpolation='none')
plt.title('heatmap_sprite')
plt.figure()
plt.imshow(ret2, interpolation='none')
plt.title('heatmap_conv')
plt.show()
sess.close()

