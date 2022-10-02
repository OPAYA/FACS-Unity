//using System.Collections;
//using System.Collections.Generic;
using UnityEngine;

public class FaceBehaviour : MonoBehaviour {
    public float speed = 500;
    public float apex = 500;
    public float intensity = 1f;

    public float talk_speed = 400;
    public bool do_talk = false;

    FaceManagerWithText fm;

    void Start() {
        fm = GetComponent<FaceManagerWithText>();
    }

    void update() {
        fm = GetComponent<FaceManagerWithText>();
        fm.ShowExpression(speed, apex, intensity);
    }
}
