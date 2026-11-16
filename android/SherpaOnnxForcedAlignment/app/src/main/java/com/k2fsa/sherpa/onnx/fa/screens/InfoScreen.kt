package com.k2fsa.sherpa.onnx.fa.screens

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.k2fsa.sherpa.onnx.VersionInfo
import com.k2fsa.sherpa.onnx.fa.currentModel
import com.k2fsa.sherpa.onnx.fa.ui.theme.Muted

private const val githubUrl = "https://github.com/k2-fsa/sherpa-onnx"
private const val docsUrl = "https://k2-fsa.github.io/sherpa/onnx/"
private const val releasesUrl = "https://github.com/k2-fsa/sherpa-onnx/releases"
private const val faModelsUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models"
private const val apkFaUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/tag/apk-fa"
private const val wechatUrl = "https://k2-fsa.github.io/sherpa/social-groups.html"

@Composable
fun InfoScreen() {
    val context = LocalContext.current

    fun openUrl(url: String) {
        context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // ===================== About =====================
        CardSection(title = "About", icon = Icons.Filled.Info) {
            Text(
                "Forced alignment: given an audio file and its transcript, " +
                    "the app computes the start and end time of every word. " +
                    "Play individual words, export a Praat TextGrid or txt " +
                    "files, and share the result with other apps.",
                fontSize = 13.sp,
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "Everything runs locally on this device — no Internet " +
                    "connection is required and no audio is uploaded anywhere.",
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
            )
        }

        // ===================== Version =====================
        CardSection(title = "Version", icon = Icons.Filled.Info) {
            InfoRow("sherpa-onnx", VersionInfo.version)
            InfoRow("Git SHA1", VersionInfo.gitSha1)
            InfoRow("Git date", VersionInfo.gitDate)
            InfoRow("onnxruntime", VersionInfo.onnxruntimeVersion)
        }

        // ===================== Current model =====================
        CardSection(title = "Current model", icon = Icons.Filled.Info) {
            Text(
                currentModel.name,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
            )
            Spacer(Modifier.height(8.dp))
            LinkRow("Download", currentModel.url) { openUrl(it) }
            LinkRow("Documentation", currentModel.doc) { openUrl(it) }
        }

        // ===================== Resources =====================
        CardSection(title = "Resources", icon = Icons.Filled.Info) {
            LinkRow("GitHub", githubUrl) { openUrl(it) }
            LinkRow("Documentation", docsUrl) { openUrl(it) }
            LinkRow("Model releases", faModelsUrl) { openUrl(it) }
            LinkRow("APK releases (apk-fa)", apkFaUrl) { openUrl(it) }
        }

        // ===================== Community =====================
        CardSection(title = "Community", icon = Icons.Filled.Info) {
            InfoRow("QQ Group", "744602236")
            LinkRow("WeChat groups", wechatUrl) { openUrl(it) }
        }

        Text(
            "Powered by sherpa-onnx",
            fontSize = 12.sp,
            color = Muted,
            textDecoration = TextDecoration.Underline,
            modifier = Modifier
                .padding(vertical = 8.dp)
                .clickable { openUrl(githubUrl) },
        )
    }
}

@Composable
private fun CardSection(
    title: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    content: @Composable () -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = icon,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                )
                Spacer(Modifier.width(8.dp))
                Text(title, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            }
            Spacer(Modifier.height(12.dp))
            Divider()
            Spacer(Modifier.height(8.dp))
            content()
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
    ) {
        Text(
            label,
            fontSize = 13.sp,
            color = Muted,
            modifier = Modifier.width(110.dp),
        )
        Text(value, fontSize = 13.sp)
    }
}

@Composable
private fun LinkRow(label: String, url: String, onClick: (String) -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick(url) }
            .padding(vertical = 6.dp, horizontal = 4.dp),
    ) {
        Column {
            Text(
                label,
                fontSize = 14.sp,
                color = MaterialTheme.colorScheme.primary,
                textDecoration = TextDecoration.Underline,
            )
            Text(
                url,
                fontSize = 11.sp,
                color = Muted,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
        }
    }
}
