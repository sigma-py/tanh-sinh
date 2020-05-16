import matplotlib.pyplot as plt
import numpy

x = numpy.linspace(-3.0, 3.0, 100)
y = numpy.tanh(numpy.sinh(x))

plt.plot(x, y, "-", color="black", linewidth=5)
plt.gca().set_aspect("equal")
plt.gca().set_axis_off()
# plt.show()
plt.savefig("logo.svg", transparent=True, bbox_inches="tight", padding=0)
