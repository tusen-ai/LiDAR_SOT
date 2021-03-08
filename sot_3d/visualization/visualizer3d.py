import numpy as np
import cv2 as cv
_JETMAP = cv.applyColorMap(np.array([i for i in range(256)]).astype(np.uint8), cv.COLORMAP_JET)
try:
    import OpenGL.GL as gl
    import pangolin
    from multiprocessing import Process, Queue
    import numpy as np
    has_pangolin = True
except ImportError:
    has_pangolin = False
    print("Import pangolin failed")
import pdb


class VisualizerPangoV2:
    BOXES3D_IDX = [(0, 1), (1, 2), (2, 3), (3, 0),
                   (0, 4), (1, 5), (2, 6), (3, 7),
                   (4, 5), (5, 6), (6, 7), (7, 4)]
    CLS2COLOR = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, 0.7, 0.0],
        [0.0, 0.7, 0.3],
        [0.0, 0.0, 1.0],
        [0.0, 0.3, 0.7],
        [0.3, 0.0, 0.7],
    ]
    @staticmethod
    def handler_pts(pts, attrs):
        if pts is None or len(pts) == 0:
            return
        if attrs:
            gl.glColor3f(*attrs['Color'])
            gl.glPointSize(attrs['PointSize'])
        else:
            gl.glColor3f(0.0, 1.0, 0.0)
            gl.glPointSize(2)
        pangolin.DrawPoints(pts)
    @staticmethod
    def handler_lines(lines, attrs):
        if lines is None:
            return
        if attrs:
            gl.glColor3f(*attrs['Color'])
            point_size = attrs['PointSize']
        else:
            gl.glColor3f(0.0, 0.0, 1.0)
            point_size = 5.0
        pangolin.DrawLine(lines, point_size=point_size)
    @staticmethod
    def handler_bboxes3d(boxes_3d, attrs):
        if boxes_3d is None:
            return
        if attrs:
            gl.glColor3f(*attrs['Color'])
            gl.glLineWidth(attrs.get('Width', 3.0))
        else:
            gl.glColor3f(0.0, 0.0, 1.0)
            gl.glLineWidth(1.0)
        gl.glBegin(gl.GL_LINES)
        for _b3d in boxes_3d:
            if type(_b3d) is tuple:
                b3d, i_cls = _b3d
                i_cls = int(i_cls) if int(i_cls) <= 5 else 5
                cls = VisualizerPangoV2.CLS2COLOR[i_cls]
                gl.glColor3f(*cls)
            else:
                b3d = _b3d
            for (ia, ib) in VisualizerPangoV2.BOXES3D_IDX:
                va = b3d[ia]
                vb = b3d[ib]
                gl.glVertex3f(*va)
                gl.glVertex3f(*vb)
        gl.glEnd()  # end of GL_LINES
        gl.glLineWidth(1.0)  # reset to 1.0
    @staticmethod
    def handler_bboxes3d_center(boxes_3d, attrs):
        if boxes_3d is None:
            return
        poses = []
        sizes = []
        for b3d in boxes_3d:
            if len(b3d) != 0:
                theta = b3d[-1]
                width, length, height = b3d[3], b3d[4], b3d[5]
                x, y, z = b3d[0], b3d[1], b3d[2]
                pose = np.identity(4)
                pose[0, 0] = np.cos(theta)
                pose[0, 1] = -np.sin(theta)
                pose[1, 0] = np.sin(theta)
                pose[1, 1] = np.cos(theta)
                pose[0, 3] = x
                pose[1, 3] = y
                pose[2, 3] = z
                poses.append(pose)
                sizes.append([width, length, height])
        if len(poses) > 0:
            if attrs:
                gl.glColor3f(*attrs['Color'])
                gl.glLineWidth(attrs.get('Width', 3.0))
            else:
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(3.0)
            pangolin.DrawBoxes(poses, sizes)

    @staticmethod
    def handler_bboxes2d(boxes_2d, attrs):
        if boxes_2d is None:
            return
        if attrs:
            gl.glColor3f(*attrs['Color'])
        else:
            gl.glColor3f(0.0, 0.0, 1.0)
        z = attrs.get('z', 0.0)
        for b in boxes_2d:
            b2d = b[0:8].reshape(4, 2)
            b3d = []
            for pt in b2d:
                x, y = pt
                b3d.append([x, y, z])
            b3d.append([0.5 * (b2d[0, 0] + b2d[1, 0]),
                        0.5 * (b2d[0, 1] + b2d[1, 1]),
                        z])
            b3d_to_draw = []
            to_draw_idx = [3, 0, 1, 2, 4, 3, 2]
            for di in to_draw_idx:
                b3d_to_draw.append(b3d[di])
            i_cls = int(b[9]) if int(b[9]) <= 5 else 5
            cls = VisualizerPangoV2.CLS2COLOR[i_cls]
            gl.glColor3f(*cls)
            pangolin.DrawLine(b3d_to_draw, point_size=int(attrs.get('PointSize', 5)))
            gl.glColor3f(*attrs['Color'])
    @staticmethod
    def handler_cam(scam, attrs):
        if attrs:
            look = attrs.get('look', [0, -70, 70])
            at = attrs.get('at', [0, 0, 0])
            up = attrs.get('up', pangolin.AxisZ)
            scam.SetModelViewMatrix(pangolin.ModelViewLookAt(look[0], look[1], look[2],
                                                             at[0], at[1], at[2],
                                                             up))
        else:
            pass
    @staticmethod
    def can_use():
        return has_pangolin
    def __init__(self, name='Main'):
        self.should_quit = False
        self.dq = Queue(maxsize=2)
        if not VisualizerPangoV2.can_use():
            self.runner = Process(target=VisualizerPangoV2.run_no_viz, args=(self.dq, self.name))
            return
        else:
            self.pcs = np.zeros(shape=(100, 3))
            self.lanes = np.zeros(shape=(100, 3))
            self.boxes = []
            self.name = name
            self.runner = Process(target=VisualizerPangoV2.run, args=(self.dq, self.name))
            self.runner.start()
    def kill(self):
        self.dq.put("quit")
    @staticmethod
    def run_no_viz(dq, name):
        should_quit = False
        while not dq.empty():
            if should_quit:
                break
            o = dq.get(timeout=1)
            if type(o) is str:
                if o == 'quit':
                    should_quit = True
                break
            else:
                continue
    # thread entry
    @staticmethod
    def run(dq, name):
        pangolin.CreateWindowAndBind(name, 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Define Projection and initial ModelView matrix
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 2000),
            pangolin.ModelViewLookAt(-70, -70, 70, 0, 0, 0, pangolin.AxisDirection.AxisZ))
        # Create Interactive View in window
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0 / 480.0)
        dcam.SetHandler(pangolin.Handler3D(scam))
        should_quit = False
        handlers = {
            'pts': VisualizerPangoV2.handler_pts,
            'boxes_3d': VisualizerPangoV2.handler_bboxes3d,
            'boxes_3d_center': VisualizerPangoV2.handler_bboxes3d_center,
            'lines': VisualizerPangoV2.handler_lines,
            'boxes_2d': VisualizerPangoV2.handler_bboxes2d
        }
        targets = {}
        for k in handlers:
            targets[k] = {}
        while not (pangolin.ShouldQuit() or should_quit):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(1.0, 1.0, 1.0, 1.0)
            dcam.Activate(scam)
            pangolin.glDrawColouredCube()
            while not dq.empty():
                o = dq.get()
                if type(o) is str:
                    if o == 'quit':
                        should_quit = True
                    break
                if type(o) is tuple:
                    op, tp, k, v = o
                    if tp == 'cam':
                        VisualizerPangoV2.handler_cam(scam, v)
                    else:
                        if op == 'reset':
                            targets[tp][k] = [v]
                        elif op == 'add':
                            if k not in targets[tp]:
                                targets[tp][k] = []
                            targets[tp][k].append(v)
            for t in handlers:
                for k in targets[t]:
                    for v in targets[t][k]:
                        handlers[t](*v)
            pangolin.FinishFrame()