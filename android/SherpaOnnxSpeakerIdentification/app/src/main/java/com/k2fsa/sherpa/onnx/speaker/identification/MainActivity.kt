package com.k2fsa.sherpa.onnx.speaker.identification

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.app.ActivityCompat
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.k2fsa.sherpa.onnx.SpeakerRecognition
import com.k2fsa.sherpa.onnx.speaker.identification.screens.HelpScreen
import com.k2fsa.sherpa.onnx.speaker.identification.screens.HomeScreen
import com.k2fsa.sherpa.onnx.speaker.identification.screens.RegisterScreen
import com.k2fsa.sherpa.onnx.speaker.identification.screens.ViewScreen
import com.k2fsa.sherpa.onnx.speaker.identification.ui.theme.SherpaOnnxSpeakerIdentificationTheme

const val TAG = "sherpa-onnx-speaker"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : ComponentActivity() {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SherpaOnnxSpeakerIdentificationTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        SpeakerRecognition.initExtractor(this.assets)
    }

    @Deprecated("Deprecated in Java")
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else {
            false
        }

        if (!permissionToRecordAccepted) {
            Log.e(TAG, "Audio record is disallowed")
            Toast.makeText(
                this,
                "This App needs access to the microphone",
                Toast.LENGTH_SHORT
            )
                .show()
            finish()
        }

        Log.i(TAG, "Audio record is permitted")
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(modifier: Modifier = Modifier) {
    val navController = rememberNavController()

    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary,
                ),
                title = {
                    Text(
                        "Next-gen Kaldi: Speaker Identification",
                        fontWeight = FontWeight.Bold,
                    )
                },
            )
        },
        content = { padding ->
            Column(Modifier.padding(padding)) {
                NavigationHost(navController = navController)

            }
        },
        bottomBar = {
            BottomNavigationBar(navController = navController)
        }
    )
}

@Composable
fun NavigationHost(navController: NavHostController) {
    NavHost(navController = navController, startDestination = NavRoutes.Home.route) {
        composable(NavRoutes.Home.route) {
            HomeScreen()
        }

        composable(NavRoutes.Register.route) {
            RegisterScreen()
        }

        composable(NavRoutes.View.route) {
            ViewScreen()
        }

        composable(NavRoutes.Help.route) {
            HelpScreen()
        }
    }
}

@Composable
fun BottomNavigationBar(navController: NavHostController) {
    NavigationBar {
        val backStackEntry by navController.currentBackStackEntryAsState()
        val currentRoute = backStackEntry?.destination?.route

        NavBarItems.BarItems.forEach { navItem ->
            NavigationBarItem(selected = currentRoute == navItem.route,
                onClick = {
                    navController.navigate(navItem.route) {
                        popUpTo(navController.graph.findStartDestination().id) {
                            saveState = true
                        }
                        launchSingleTop = true
                        restoreState = true
                    }
                },
                icon = {
                    Icon(imageVector = navItem.image, contentDescription = navItem.title)
                }, label = {
                    Text(text = navItem.title)
                })
        }
    }
}

@Preview(showBackground = true)
@Composable
fun MainScreenPreview() {
    SherpaOnnxSpeakerIdentificationTheme {
        MainScreen()
    }
}