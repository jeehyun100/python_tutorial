import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#plt.interactive(True)
buf = plt.imread('q1.png')
#buf = cv2.imread('q1.png')

tmp_r = np.zeros(buf.shape, dtype=buf.dtype)
tmp_g = np.zeros(buf.shape, dtype=buf.dtype)
tmp_b = np.zeros(buf.shape, dtype=buf.dtype)
#tmp_im[:, :, c] = im[:, :, c]
#

buf_bgr = np.zeros(buf.shape, dtype=buf.dtype)

tmp_r[:,:,0] = buf[:,:,0]
tmp_g[:,:,1] = buf[:,:,1]
tmp_b[:,:,2] = buf[:,:,2]

buf_bgr[:,:,0] = buf[:,:,1]
buf_bgr[:,:,1] = buf[:,:,2]
buf_bgr[:,:,2] = buf[:,:,0]


if tmp_r.dtype != np.uint8:
    tmp_r = (255*tmp_r).astype(np.uint8)
if tmp_g.dtype != np.uint8:
    tmp_g = (255*tmp_g).astype(np.uint8)
if tmp_b.dtype != np.uint8:
    tmp_b = (255*tmp_b).astype(np.uint8)
R_max = tmp_r[:,:,0].max()
R_arg = tmp_r[:,:,0].mean()


G_max = tmp_g[:,:,1].max()
G_arg = tmp_g[:,:,1].mean()

B_max = tmp_b[:,:,2].max()
B_arg = tmp_b[:,:,2].mean()

r_max_anno = "R_max : {0}".format(R_max)
r_arg_anno = "R_arg : {0}".format(round(R_arg,2))
g_max_anno = "G_max : {0}".format(G_max)
g_arg_anno = "G_arg : {0}".format(round(G_arg,2))
b_max_anno = "B_max : {0}".format(B_max)
b_arg_anno = "B_arg : {0}".format(round(B_arg,2))


#fig, ((ax1,ax2), (ax3,ax4),(ax5, ax6)) = plt.subplots(3, 2)

fig = plt.figure(figsize=(7,7))
#(ax1, ax2), (ax3, ax4) = axs
fig.suptitle('Problem 2')


ax1 = fig.add_subplot(3,2,1)
ax1.imshow(buf)
ax1.title.set_text('Original')
ax1.set_xticks([])
ax1.set_yticks([])

bbox_props_f = dict(boxstyle='round', fc='w', lw=2)

ax2 = fig.add_subplot(3,2,2)
ax2.imshow(tmp_r)
ax2.title.set_text('Red')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate(r_max_anno, xy=(170,80), bbox=bbox_props_f)
ax2.annotate(r_arg_anno, xy=(170,160), bbox=bbox_props_f)

ax3 = fig.add_subplot(3,2,3)
ax3.imshow(tmp_g)
ax3.title.set_text('Green')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.annotate(g_max_anno, xy=(170,80), bbox=bbox_props_f)
ax3.annotate(g_arg_anno, xy=(170,160), bbox=bbox_props_f)

ax4 = fig.add_subplot(3,2,4)
ax4.imshow(tmp_b)
ax4.title.set_text('Blue')
ax4.set_xticks([])
ax4.set_yticks([])
ax4.annotate(b_max_anno, xy=(170,80), bbox=bbox_props_f)
ax4.annotate(b_arg_anno, xy=(170,160), bbox=bbox_props_f)
# for ax in axs.flat:
#     ax.label_outer()
#
# plt.imshow(tmp_r)
ax5 = fig.add_subplot(3,2,5)
ax5.title.set_text('Convert G B R')
ax5.imshow(buf_bgr)
ax5.set_xticks([])
ax5.set_yticks([])


plt.show()



#
# def split_into_rgb_channels(image):
#   '''Split the target image into its red, green and blue channels.
#   image - a numpy array of shape (rows, columns, 3).
#   output - three numpy arrays of shape (rows, columns) and dtype same as
#            image, containing the corresponding channels.
#   '''
#   red = image[:,:,2]
#   green = image[:,:,1]
#   blue = image[:,:,0]
#   return red, green, blue
#
# #[b,g,r]=np.dsplit(buf,buf.shape[-1])
# #
# # # USING OPENCV SPLIT FUNCTION
# #b,g,r=cv2.split(buf)

#
# red, green, blue = split_into_rgb_channels(buf)
# img = np.zeros((values.shape[0], values.shape[1], 3), dtype=values.dtype)
# img[:, :, channel] = values




# #print img
# for values, color, channel in zip((red, green, blue), ('red', 'green', 'blue'), (2,1,0)):
#     img = np.zeros((values.shape[0], values.shape[1], 3),dtype = values.dtype)
#     img[:,:,channel] = values
#     #print "Saving Image: {}.".format(name+color+ext)
#     while True:
#       cv2.imshow('test', img)
#       # print('output {}'.format(b))
#       if cv2.waitKey(25) & 0xFF == ord('q'):
#           cv2.destroyAllWindows()
#           break
#           #cv2.imwrite(os.path.join(dirname, name+color+ext), img)
#
# print("dfada")
#plt.show()




