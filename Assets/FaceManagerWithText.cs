using System.Collections.Generic;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Text;

[RequireComponent(typeof(SkinnedMeshRenderer))]
public class FaceManagerWithText : MonoBehaviour {
    public float speed = 500;
    public float apex = 500;
    public float intensity = 1f;

    public float talk_speed = 400;
    public bool do_talk = false;

    FaceManagerWithText fm;
   // FaceExpression current_expression;
    float current_expression_step = 0;
    float current_expression_duration = 0;
    float current_expression_duration_inv = 0;
    float current_expression_rest = 0;
    float last_expression_intensity = 0;
    float current_expression_intensity = 0;
    float speech_duration = 0;
    float speech_duration_inv = 0;
    float speech_step = 0;
    List<(float, float, float)> stack = new List<(float, float, float)>();
    float[] movstartweights = new float[nAUs] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] maxweights = new float[nAUs] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] speech_movstartweights = new float[nAUs] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] speech_maxweights = new float[nAUs] {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool showing_expression = false;
    bool speaking = false;

    SkinnedMeshRenderer skinnedMeshRenderer;
    int blendShapeCount;

    const int nAUs = 20;

    [HideInInspector] public bool isTxStarted = false;
    [SerializeField] Texture _image = null;
    [SerializeField] string IP = "127.0.0.1"; // local host
    [SerializeField] int rxPort = 8000; // port to receive data from Python on
    [SerializeField] int txPort = 8001; // port to send data to Python on

    int i = 0; // DELETE THIS: Added to show sending data from Unity to Python via UDP
    string text = "0";
    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

        // "AU01",
        // "AU02",
        // "AU04",
        // "AU06",
        // "AU07",
        // "AU10",
        // "AU12",
        // "AU14",
        // "AU15",
        // "AU17",
        // "AU23",
        // "AU24",
        
        /***
        AU4 5
        AU1 1
        AU2 3
        AU5
        AU7 9
        AU6 7
        AU9
        AU10 11
        AU17 17
        AU15 15
        AU25
        AU26
        AU27
        AU16
        AU20
        AU12 13
        AU23 19
        AU24 21
        ***/
    void Start()
    {
        ShowExpression(speed, apex, intensity);
    }
    private void ProcessInput(string input)
    {
        // PROCESS INPUT RECEIVED STRING HERE

        if (!isTxStarted) // First data arrived so tx started
        {
            isTxStarted = true;
        }
    }

    // Receive data, update packets received
    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                
                print(">> " + text[0]);
                print(">> " + text[1]);
                print(">> " + text[2]);
                print(">> " + text[3]);
                ProcessInput(text);
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    public void ShowExpression(float speed, float rest, float intensity) {
        if (showing_expression) {
            stack.Add((speed, rest, intensity));
            return;
        }

        movstartweights = (float[])maxweights.Clone();
        last_expression_intensity = last_expression_intensity == 0 ? intensity : current_expression_intensity;
        // maxweights = GetWeights(expression);
        text = GetData();

        float AU1 = float.Parse(text.Split(',')[0]);
        float AU2 = float.Parse(text.Split(',')[1].Substring(1));
        float AU4 = float.Parse(text.Split(',')[2].Substring(1));
        float AU5 = float.Parse(text.Split(',')[3].Substring(1));
        float AU6 = float.Parse(text.Split(',')[4].Substring(1));
        float AU7 = float.Parse(text.Split(',')[5].Substring(1));
        float AU9 = float.Parse(text.Split(',')[6].Substring(1));
        float AU10 = float.Parse(text.Split(',')[7].Substring(1));
        float AU11 = float.Parse(text.Split(',')[8].Substring(1));
        float AU12 = float.Parse(text.Split(',')[9].Substring(1));
        float AU14 = float.Parse(text.Split(',')[10].Substring(1));
        float AU15 = float.Parse(text.Split(',')[11].Substring(1));
        float AU17 = float.Parse(text.Split(',')[12].Substring(1));
        float AU20 = float.Parse(text.Split(',')[13].Substring(1));
        float AU23 = float.Parse(text.Split(',')[14].Substring(1));
        float AU24 = float.Parse(text.Split(',')[15].Substring(1));
        float AU25 = float.Parse(text.Split(',')[16].Substring(1));
        float AU26 = float.Parse(text.Split(',')[17].Substring(1));
        float AU28 = float.Parse(text.Split(',')[18].Substring(1));
        float AU43 = float.Parse(text.Split(',')[19].Substring(1));
    
        maxweights = new float[nAUs]{AU5, AU12, AU28, AU23, AU24, 0, 0, 0, 0, AU43, AU26, AU4, AU6, AU10, AU15, AU9, AU17, AU2, AU1, AU14};
        maxweights = new float[nAUs]{0, AU12, 0, AU23, AU24, 0, 0, 0, 0, 0, 0, AU4, AU6, AU10, AU15, 0, AU17, AU2, AU1, AU14};
     
        current_expression_step = 0;
        current_expression_duration = speed;
        current_expression_duration_inv = 1 / current_expression_duration;
        current_expression_rest = rest;
        current_expression_intensity = intensity;
        showing_expression = true;
    }

   
    float SmoothFunctionExpression(float t) {
        return -2*t*t*t + 3*t*t;
    }

    public string GetData(){
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        byte[] data = client.Receive(ref anyIP);
        string text = Encoding.UTF8.GetString(data);
        return text;
    }

    public void SendData(string message)
    {
        byte[] data = Encoding.UTF8.GetBytes(message);
        client.Send(data, data.Length, remoteEndPoint);
    }

    void Awake() {
        skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
        blendShapeCount = skinnedMeshRenderer.sharedMesh.blendShapeCount;

        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Create local client
        client = new UdpClient(rxPort);
    }

    void Update() {

        ShowExpression(speed, apex, intensity);
       
        if (showing_expression) {
            if (current_expression_step > current_expression_duration + current_expression_rest) {
                showing_expression = false;
            } else {
                // During rising movement
                float t;
                if (current_expression_step <= current_expression_duration) {
                    for (int i = 0; i < blendShapeCount; i++) {
                        t = maxweights[i];
                     
                        skinnedMeshRenderer.SetBlendShapeWeight(i, 100*SmoothFunctionExpression(Mathf.Clamp(t, 0, 1)));
                    }
                }
                current_expression_step += 1000000*Time.deltaTime;
            }
        } 
        
        else {
            if (stack.Count > 0) {
                (float, float, float) cached = stack[0];
                ShowExpression(cached.Item1, cached.Item2, cached.Item3);
                stack.RemoveAt(0);
            } 
            else {
                ShowExpression(speed, 0, 1);
            }
        }
    }
}
