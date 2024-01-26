using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandTracking : MonoBehaviour
{
    public UDPReceive UDPReceive;
    public GameObject[] HandPoints;
    public GameObject Hand;
    
    public GameObject Camera;

    // Start is called before the first frame update
    void Start()
    {
        Camera = GameObject.Find("Main Camera");
    }

    // Update is called once per frame
    void Update()
    {

        string data = UDPReceive.data;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length-1, 1);
        print(data);
        string[] points = data.Split(',');
        print(points[0]);



        // x1,y1,z1,x2,y2,z2,...
        for (int i = 0; i < 21; i++){
            float x = 5-float.Parse(points[i*3])/100;
            float y = float.Parse(points[i*3+1])/100;
            float z = float.Parse(points[i*3+2])/100;
            HandPoints[i].transform.localPosition = new Vector3(x-Camera.transform.position.x,y-Camera.transform.position.y,z-Camera.transform.position.z);
        }
        
    }
}
