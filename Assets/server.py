import time
import math

from feat import Detector
from feat.utils import read_pictures

import cv2
import numpy as np

from skimage.feature import hog
from scipy.spatial import ConvexHull
from skimage.morphology.convex_hull import grid_points_in_poly


class UdpComms():
    def __init__(self,udpIP,portTX,portRX,enableRX=False,suppressWarnings=True):
        """
        Constructor
        :param udpIP: Must be string e.g. "127.0.0.1"
        :param portTX: integer number e.g. 8000. Port to transmit from i.e From Python to other application
        :param portRX: integer number e.g. 8001. Port to receive on i.e. From other application to Python
        :param enableRX: When False you may only send from Python and not receive. If set to True a thread is created to enable receiving of data
        :param suppressWarnings: Stop printing warnings if not connected to other application
        """

        import socket

        self.udpIP = udpIP
        self.udpSendPort = portTX
        self.udpRcvPort = portRX
        self.enableRX = enableRX
        self.suppressWarnings = suppressWarnings # when true warnings are suppressed
        self.isDataReceived = False
        self.dataRX = None

        # Connect via UDP
        self.udpSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # internet protocol, udp (DGRAM) socket
        self.udpSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # allows the address/port to be reused immediately instead of it being stuck in the TIME_WAIT state waiting for late packets to arrive.
        self.udpSock.bind((udpIP, portRX))

        # Create Receiving thread if required
        if enableRX:
            import threading
            self.rxThread = threading.Thread(target=self.ReadUdpThreadFunc, daemon=True)
            self.rxThread.start()

    def __del__(self):
        self.CloseSocket()

    def CloseSocket(self):
        # Function to close socket
        self.udpSock.close()

    def SendData(self, strToSend):
        # Use this function to send string to C#
        self.udpSock.sendto(bytes(strToSend,'utf-8'), (self.udpIP, self.udpSendPort))

    def ReceiveData(self):
        """
        Should not be called by user
        Function BLOCKS until data is returned from C#. It then attempts to convert it to string and returns on successful conversion.
        An warning/error is raised if:
            - Warning: Not connected to C# application yet. Warning can be suppressed by setting suppressWarning=True in constructor
            - Error: If data receiving procedure or conversion to string goes wrong
            - Error: If user attempts to use this without enabling RX
        :return: returns None on failure or the received string on success
        """
        if not self.enableRX: # if RX is not enabled, raise error
            raise ValueError("Attempting to receive data without enabling this setting. Ensure this is enabled from the constructor")

        data = None
        try:
            data, _ = self.udpSock.recvfrom(1024)
            data = data.decode('utf-8')
        except WindowsError as e:
            if e.winerror == 10054: # An error occurs if you try to receive before connecting to other application
                if not self.suppressWarnings:
                    print("Are You connected to the other application? Connect to it!")
                else:
                    pass
            else:
                raise ValueError("Unexpected Error. Are you sure that the received data can be converted to a string")

        return data

    def ReadUdpThreadFunc(self): # Should be called from thread
        """
        This function should be called from a thread [Done automatically via constructor]
                (import threading -> e.g. udpReceiveThread = threading.Thread(target=self.ReadUdpNonBlocking, daemon=True))
        This function keeps looping through the BLOCKING ReceiveData function and sets self.dataRX when data is received and sets received flag
        This function runs in the background and updates class variables to read data later

        """

        self.isDataReceived = False # Initially nothing received

        while True:
            data = self.ReceiveData()  # Blocks (in thread) until data is returned (OR MAYBE UNTIL SOME TIMEOUT AS WELL)
            self.dataRX = data # Populate AFTER new data is received
            self.isDataReceived = True
            # When it reaches here, data received is available

    def ReadReceivedData(self):
        """
        This is the function that should be used to read received data
        Checks if data has been received SINCE LAST CALL, if so it returns the received string and sets flag to False (to avoid re-reading received data)
        data is None if nothing has been received
        :return:
        """

        data = None

        if self.isDataReceived: # if data has been received
            self.isDataReceived = False
            data = self.dataRX
            self.dataRX = None # Empty receive buffer

        return data

def extract_hog(
        frame,
        orientation=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    ):
        """Extract HOG features from a SINGLE frame.
        Args:
            frame (array]): Frame of image]
            orientation (int, optional): Orientation for HOG. Defaults to 8.
            pixels_per_cell (tuple, optional): Pixels per cell for HOG. Defaults to (8,8).
            cells_per_block (tuple, optional): Cells per block for HOG. Defaults to (2,2).
            visualize (bool, optional): Whether to provide the HOG image. Defaults to False.
        Returns:
            hog_output: array of HOG features, and the HOG image if visualize is True.
        """

        hog_output = hog(
            frame,
            orientations=orientation,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=visualize,
            channel_axis=-1,
        )
        if visualize:
            return (hog_output[0], hog_output[1])
        else:
            return hog_output

def align_face_68pts(img, img_land, box_enlarge, img_size=112):
    """Performs affine transformation to align the images by eyes.
    Performs affine alignment including eyes.
    Args:
        img: gray or RGB
        img_land: 68 system flattened landmarks, shape:(136)
        box_enlarge: relative size of face on the image. Smaller value indicate larger proportion
        img_size = output image size
    Return:
        aligned_img: the aligned image
        new_land: the new landmarks
    """
    leftEye0 = (
        img_land[2 * 36]
        + img_land[2 * 37]
        + img_land[2 * 38]
        + img_land[2 * 39]
        + img_land[2 * 40]
        + img_land[2 * 41]
    ) / 6.0
    leftEye1 = (
        img_land[2 * 36 + 1]
        + img_land[2 * 37 + 1]
        + img_land[2 * 38 + 1]
        + img_land[2 * 39 + 1]
        + img_land[2 * 40 + 1]
        + img_land[2 * 41 + 1]
    ) / 6.0
    rightEye0 = (
        img_land[2 * 42]
        + img_land[2 * 43]
        + img_land[2 * 44]
        + img_land[2 * 45]
        + img_land[2 * 46]
        + img_land[2 * 47]
    ) / 6.0
    rightEye1 = (
        img_land[2 * 42 + 1]
        + img_land[2 * 43 + 1]
        + img_land[2 * 44 + 1]
        + img_land[2 * 45 + 1]
        + img_land[2 * 46 + 1]
        + img_land[2 * 47 + 1]
    ) / 6.0
    deltaX = rightEye0 - leftEye0
    deltaY = rightEye1 - leftEye1
    l = math.sqrt(deltaX * deltaX + deltaY * deltaY)
    sinVal = deltaY / l
    cosVal = deltaX / l
    mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
    mat2 = np.mat(
        [
            [leftEye0, leftEye1, 1],
            [rightEye0, rightEye1, 1],
            [img_land[2 * 30], img_land[2 * 30 + 1], 1],
            [img_land[2 * 48], img_land[2 * 48 + 1], 1],
            [img_land[2 * 54], img_land[2 * 54 + 1], 1],
        ]
    )
    mat2 = (mat1 * mat2.T).T
    cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
    cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
    if float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(
        max(mat2[:, 1]) - min(mat2[:, 1])
    ):
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
    else:
        halfSize = 0.5 * box_enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
    scale = (img_size - 1) / 2.0 / halfSize
    mat3 = np.mat(
        [
            [scale, 0, scale * (halfSize - cx)],
            [0, scale, scale * (halfSize - cy)],
            [0, 0, 1],
        ]
    )
    mat = mat3 * mat1
    aligned_img = cv2.warpAffine(
        img,
        mat[0:2, :],
        (img_size, img_size),
        cv2.INTER_LINEAR,
        borderValue=(128, 128, 128),
    )
    land_3d = np.ones((int(len(img_land) / 2), 3))
    land_3d[:, 0:2] = np.reshape(np.array(img_land), (int(len(img_land) / 2), 2))
    mat_land_3d = np.mat(land_3d)
    new_land = np.array((mat * mat_land_3d.T).T)
    new_land = np.array(list(zip(new_land[:, 0], new_land[:, 1]))).astype(int)

    return aligned_img, new_land

def extract_face(frame, detected_faces, landmarks, size_output=112):
        """Extract a face in a frame with a convex hull of landmarks.
        This function extracts the faces of the frame with convex hulls and masks out the rest.
        Args:
            frame (array): The original image]
            detected_faces (list): face bounding box
            landmarks (list): the landmark information]
            size_output (int, optional): [description]. Defaults to 112.
        Returns:
            resized_face_np: resized face as a numpy array
            new_landmarks: landmarks of aligned face
        """
        detected_faces = np.array(detected_faces)
        landmarks = np.array(landmarks)

        detected_faces = detected_faces.astype(int)

        aligned_img, new_landmarks = align_face_68pts(
            frame, landmarks.flatten(), 2.5, img_size=size_output
        )

        hull = ConvexHull(new_landmarks)
        mask = grid_points_in_poly(
            shape=np.array(aligned_img).shape,
            # for some reason verts need to be flipped
            verts=list(
                zip(
                    new_landmarks[hull.vertices][:, 1],
                    new_landmarks[hull.vertices][:, 0],
                )
            ),
        )
        mask[
            0 : np.min([new_landmarks[0][1], new_landmarks[16][1]]),
            new_landmarks[0][0] : new_landmarks[16][0],
        ] = True
        aligned_img[~mask] = 0
        resized_face_np = aligned_img
        resized_face_np = cv2.cvtColor(resized_face_np, cv2.COLOR_BGR2RGB)

        return (
            resized_face_np,
            new_landmarks,
        )  # , hull, mask, np.array(ali

def concatenate_batch(indexed_length, au_results):
    """
    NEW
    helper function to convert batch AUs to desired list of list
    only useful for our emotion and au prediction results
    Args:
        indexed_length: (list) the list index for number of faces in each frame.
                        if you have 2 faces in each frame and you batch process 4
                        frames, it will be [2,2,2,2]
        au_results: (np.array), immediate result from running our
                    au/emotion models
    Returns:
        list_concat: (list of list). The list which contains the number of faces. for example
        if you process 2 frames and each frame contains 4 faces, it will return:
            [[xxx,xxx,xxx,xxx],[xxx,xxx,xxx,xxx]]
    """
    list_concat = []
    new_lens = np.insert(np.cumsum(indexed_length), 0, 0)
    for ij in range(len(indexed_length)):
        list_concat.append(au_results[new_lens[ij] : new_lens[ij + 1], :])
    return list_concat

def batch_hog(frames, detected_faces, landmarks):
    """
    NEW
    Helper function used in batch processing hog features
    frames is a batch of frames
    """

    len_index = [len(aa) for aa in landmarks]
    lenth_cumu = np.cumsum(len_index)
    lenth_cumu2 = np.insert(lenth_cumu, 0, 0)
    new_lands_list = []
    flat_faces = [item for sublist in detected_faces for item in sublist]
    flat_land = [item for sublist in landmarks for item in sublist]
    hogs_arr = None

    for i in range(len(flat_land)):

        frame_assignment = np.where(i < lenth_cumu)[0][0]

        convex_hull, new_lands = extract_face(
            frame=frames[frame_assignment],
            detected_faces=[flat_faces[i][0:4]],
            landmarks=flat_land[i],
            size_output=112,
        )
        hogs = extract_hog(frame=convex_hull, visualize=False).reshape(1, -1)
        if hogs_arr is None:
            hogs_arr = hogs
        else:
            hogs_arr = np.concatenate([hogs_arr, hogs], 0)

        new_lands_list.append(new_lands)

    new_lands = []
    for i in range(len(lenth_cumu)):
        new_lands.append(new_lands_list[lenth_cumu2[i] : (lenth_cumu2[i + 1])])

    return (hogs_arr, new_lands)

if __name__ == '__main__':
    
    labels = ['AU01', 'AU02', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU23', 'AU24']

    face_model = 'faceboxes'
    landmark_model="pfld"
    au_model="logistic"
    emotion_model="resmasknet"
    facepose_model="img2pose"

    detector = Detector(
        face_model=face_model,
        landmark_model=landmark_model,
        au_model=au_model,
        emotion_model='svm',
        facepose_model='pnp',
    )

    cam = cv2.VideoCapture(1)

    sock = UdpComms(udpIP="127.0.0.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
    
    detected_faces = [[[787, 458, 1236, 983, 0.999]]]
    x, y, w, d = 787, 458, 1236, 983
  
    i = 0
    while True: 
        if i == 0:
            time.sleep(2)
        check, frame = cam.read()
        if check:
            model_start_time = time.time()
            frame = np.expand_dims(frame, 0)
            start_time = time.time()
           
            if detected_faces is not None:
                start_time = time.time()
                
                detected_landmarks = detector.detect_landmarks(frame, detected_faces)
                
                start_time = time.time()
                hog_arr, new_lands = batch_hog(
                    frames=frame, detected_faces=detected_faces, landmarks=detected_landmarks
                )
                
                detected_au = detector.detect_aus(hog_arr, new_lands)[0]

                au_list = []
                for au in detected_au:
                    au = str(round(au, 3))

                    if len(au) == 5:
                        au_list.append(au)
                    elif len(au) <= 5:
                        diff = 5 - len(au)
                        for ze in range(diff):
                            au += '0'
                        au_list.append(au)


                print(str(au_list), time.time() - model_start_time)
                au_list = str(au_list)[1:-1].replace("'", "")
                sock.SendData(au_list) # Send this string to other application
            i += 1
            
        # cam.release()
        # cv2.destroyAllWindows()